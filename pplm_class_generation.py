from transformers.generation_utils import GenerationMixin
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers import top_k_top_p_filtering
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor, MinLengthLogitsProcessor

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import torch.nn.functional as F
import random

SMALL_CONST = 1e-15
BIG_CONST = 1e10





def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)


class PPLMGeneration(GenerationMixin):
    def __init__(self, model, classifier, device):
        self.model = model
        self.classifier = classifier
        self.device = device
        
        # freeze parameters in model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # send model to device
        self.model.to(self.device)
        self.model.eval()
        self.classifier.to(self.device)
        self.classifier.eval()
        
        # components using in perturb_past
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')  # you need to set your reduction as 'sum'
                                                                   # because you don't update your model parameter, you update your Key Value matrix 
                                                                   # which is very dynamic to number of return samples (batch size)                                                          
        self.wte = self.model.resize_token_embeddings()
        
    
    def __call__(self):
        pass
    
    
    def generate(self,
                 input_ids=None,
                 max_length=None,
                 min_length=None,
                 do_sample=None,
                 num_beams=1,
                 temperature=None,
                 top_k=None,
                 top_p=None,
                 repetition_penalty=None,
                 bos_token_id=None,
                 pad_token_id=None,
                 eos_token_id=None,
                 no_repeat_ngram_size=None,
                 num_return_sequences=None,
                 use_cache=None,
                 perturb=True,
                 class_label=None,
                 horizon_length=None,
                 window_length=None,
                 decay=None,
                 gamma=None,
                 kl_scale=None,
                 gm_scale=None,
                 stepsize=None,
                 num_iterations=None,
                 grad_length=None,
                 **model_kwargs):
        
        num_beams = num_beams if num_beams is not None else self.model.config.num_beams
        max_length = max_length if max_length is not None else self.model.config.max_length
        do_sample = do_sample if do_sample is not None else self.model.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.model.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.model.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.model.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id

        # skip when input ids is none

        if model_kwargs.get("attention_mask", None) is None:
            # init 'attention_mask' depending on 'pad_token_id'
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )


        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        # skip everything about encoder-decoder model

        if input_ids.shape[-1] >= max_length:
            print('you should set your seq len smaller than max length')

        # determine generation mode
        is_greedy_gen_mode = (num_beams == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and do_sample is True

        # set model kwargs 
        model_kwargs['use_cache'] = use_cache

        # get distribution pre processing samplers
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.model.config.repetition_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.model.config.no_repeat_ngram_size
        )
        min_length = min_length if min_length is not None else self.model.config.min_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id
        
        logits_processor = LogitsProcessorList()
        if repetition_penalty is not None and repetition_penalty != 1.0:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            logits_processor.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if min_length is not None and eos_token_id is not None and min_length > -1:
            logits_processor.append(MinLengthLogitsProcessor(min_length, eos_token_id))


        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length,
            max_time=None
        )

        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(f'num_return_sequences has to be 1, but you putted {num_return_sequences}')

            # greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs
            )

        elif is_sample_gen_mode:
            # get probability distribution warper
            # logits_warper = self._get_logits_warper(
            #     top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            # )

            # expand input_ids with 'num_return_sequences' additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                **model_kwargs,
            )

            # sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                # logits_warper=logits_warper,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                stopping_criteria=stopping_criteria,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                perturb=perturb,
                class_label=class_label,
                horizon_length=horizon_length,
                window_length=window_length,
                decay=decay,
                gamma=gamma,
                kl_scale=kl_scale,
                gm_scale=gm_scale,
                stepsize=stepsize,
                num_iterations=num_iterations,
                grad_length=grad_length,
                **model_kwargs,
            )


    def greedy_search(self,
                      input_ids,
                      logits_processor=None,
                      stopping_criteria=None,
                      max_length=None,
                      pad_token_id=None,
                      eos_token_id=None,
                      **model_kwargs,
                      ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = max_length if max_length is not None else self.model.config.max_length
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.model.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id

        # init sequence length tensors
        sequence_lengths, unfinished_squences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )

        while cur_len < max_length:
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            pass
        return 'hello'



    def sample(self,
               input_ids,
               logits_processor=None,
               stopping_criteria=None,
            #    logits_warper=None,
               top_k=None,
               top_p=None,
               temperature=None,
               max_length=None,
               pad_token_id=None,
               eos_token_id=None,
               perturb=True,
               class_label=None,
               horizon_length=None,
               window_length=None,
               decay=None,
               gamma=None,
               kl_scale=None,
               gm_scale=None,
               stepsize=None,
               num_iterations=None,
               grad_length=None,
               **model_kwargs,
               ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = max_length if max_length is not None else self.model.config.max_length
        validate_stopping_criteria(stopping_criteria, max_length)
        # logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.model.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id


        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )

        # auto-regressive generation
        while cur_len < max_length:
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            next_token_logits, past, unpert_logits, unpert_probs = self.get_next_token_logits(cur_len,
                                                                                              model_inputs,
                                                                                              perturb,
                                                                                              class_label=class_label,
                                                                                              horizon_length=horizon_length,
                                                                                              window_length=window_length,
                                                                                              decay=decay,
                                                                                              gamma=gamma,
                                                                                              kl_scale=kl_scale,
                                                                                              stepsize=stepsize,
                                                                                              num_iterations=num_iterations,
                                                                                              grad_length=grad_length)
            next_token_logits_last = next_token_logits[:, -1, :] / temperature
            next_token_score = logits_processor(input_ids, next_token_logits_last)
            pert_probs = F.softmax(next_token_score, dim=-1)
            
            if perturb:
                unpert_next_token_logits_last = unpert_logits[:, -1, :] / temperature
                unpert_next_token_score = logits_processor(input_ids, unpert_next_token_logits_last)
                unpert_probs = F.softmax(unpert_next_token_score, dim=-1)
                pert_probs = (pert_probs ** gm_scale) * (unpert_probs ** (1 - gm_scale))
                
            # pert_probs = top_k_top_p_filtering(logits=pert_probs, top_k=top_k, top_p=top_p)
            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True) 
            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

            # sample
            next_tokens = torch.multinomial(pert_probs, num_samples=1).squeeze(1)

             # add code that transforms next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1
            
            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # stop when there is a </s> in each sentence, or if we exceed the maximum length
            if unfinished_sequences.max() == 0:
                break

            if stopping_criteria(input_ids, None):
                break

            outputs = BaseModelOutputWithPast(past_key_values=past)

            # update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )
        
        return input_ids



    def get_next_token_logits(self,
                              cur_len,
                              model_inputs,
                              perturb=True,
                              class_label=None,
                              horizon_length=1,
                              window_length=0,
                              decay=False,
                              gamma=1.5,
                              kl_scale=0.01,
                              stepsize=0.02,
                              num_iterations=3,
                              grad_length=10000,
                              ):
        if model_inputs['past_key_values'] is None and model_inputs['input_ids'] is not None:
            last = model_inputs['input_ids'][:,-1:]
            if model_inputs['input_ids'].shape[1] > 1:
                _, past, _ = self.model(model_inputs['input_ids'][:,:-1])[:]
        else:
            past = model_inputs['past_key_values']
            last = model_inputs['input_ids']
        
        unpert_logits, unpert_past, unpert_all_hidden = self.model(model_inputs['input_ids'])[:]
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if grad max length
        if cur_len >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past
        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, unpert_probs = self.perturb_past(
                    past,
                    self.model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    stepsize=current_stepsize,
                    classifier=self.classifier,
                    class_label=class_label,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=self.device
                )
            else:
                pert_past = past
        
        pert_logits, past, _ = self.model(last, past_key_values=pert_past)[:]
        return pert_logits, past, unpert_logits, unpert_probs



    def perturb_past(self,
                     past,
                     model,
                     last,
                     unpert_past=None,
                     unpert_logits=None,
                     accumulated_hidden=None,
                     stepsize=0.01,
                     classifier=None,
                     class_label=None,
                     num_iterations=3,
                     horizon_length=1,
                     window_length=0,
                     decay=False,
                     gamma=1.5,
                     kl_scale=0.01,
                     device='cpu'):
        # accumulator only save grad number (numpy)
        grad_accumulator = tuple([
            (np.zeros(p[0].shape).astype("float32"),
             np.zeros(p[1].shape).astype("float32"))
            for p in past
        ])
        
        if accumulated_hidden is None:
            accumulated_hidden = 0
            
        window_mask = self.get_window_mask(decay, window_length, past, device)

        # unpert probs saved for kl div loss
        unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
        unpert_probs_correct = unpert_probs + SMALL_CONST * (unpert_probs <= SMALL_CONST).float().to(device).detach()
        
        new_accumulated_hidden = None
        for i in range(num_iterations):
            # copy grad_accumulated data into torch tensor
            curr_perturbation = tuple([
                (torch.tensor(p_[0], requires_grad=True, device=device),
                 torch.tensor(p_[1], requires_grad=True, device=device))
                for p_ in grad_accumulator
            ])
            
            # add curr_perturbation and past 
            perturbed_past = tuple([
                (p_[0] + past[idx][0], p_[1] + past[idx][1])
                for idx, p_ in enumerate(curr_perturbation)
            ])
            _, _, curr_length, _ = curr_perturbation[0][0].shape
            all_logits, _, all_hidden = model(last, past_key_values=perturbed_past)[:]
            new_accumulated_hidden = accumulated_hidden + torch.sum(all_hidden[-1], dim=1).detach()
            logits = all_logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            loss = 0.0
        
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            
            for _ in range(horizon_length):
                curr_unpert_past, new_accumulated_hidden = self.get_new_accumulated_hidden(model, curr_probs, curr_unpert_past, new_accumulated_hidden, self.wte)
            
            prediction = classifier(new_accumulated_hidden / (curr_length + 1 + horizon_length)) # need to deal with lstm
            label = torch.tensor(prediction.shape[0]*[class_label], device=device, dtype=torch.long)
            discrim_loss = self.loss_fn(prediction, label) 
            loss += discrim_loss
            
            if kl_scale > 0.0:
                kl_loss = self.get_kl_loss(kl_scale, unpert_probs_correct, probs, device)
                loss += kl_loss
                
            loss.backward()
            
            # normalize and add to grad_accumulator at once
            grad_accumulator = self.add_grad_to_accumulator(window_mask, stepsize, gamma, curr_perturbation, grad_accumulator)

        pert_past = tuple([
            (past[idx][0] + torch.tensor(p_[0], requires_grad=False, device=device),
             past[idx][1] + torch.tensor(p_[1], requires_grad=False, device=device))
             for idx, p_ in enumerate(grad_accumulator)
        ])
        
        return pert_past, unpert_probs
            
        
    def get_window_mask(self, decay, window_length, past, device):
        assert past[0][0].shape == past[0][1].shape, 'past key value shape should be same'
        if decay:
            decay_mask = torch.arange(0.0, 1.0 + SMALL_CONST, 1.0 / (window_length))[1:]
        else:
            decay_mask = 1.0
            
        _, _, curr_length, _ = past[0][0].shape
        
        if curr_length > window_length and window_length > 0:
            ones_key_val_shape = tuple(past[0][0].shape[:-2]) + tuple([window_length]) + tuple(past[0][0].shape[-1:])
            zeros_key_val_shape = tuple(past[0][0].shape[:-2]) + tuple([curr_length - window_length]) + tuple(past[0][0].shape[-1:])
            ones_mask = torch.ones(ones_key_val_shape)
            ones_mask = decay_mask * ones_mask.permute(0,1,3,2)
            ones_mask = ones_mask.permute(0,1,3,2)
            window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).to(device)
        else:
            window_mask = torch.ones_like(past[0][0]).to(device)
        return window_mask
    
    
    def add_grad_to_accumulator(self,
                                window_mask,
                                stepsize,
                                gamma,
                                curr_perturbation,
                                grad_accumulator):
        grad_norms = lambda x : torch.norm(x.grad * window_mask) + SMALL_CONST
        grad = lambda x : -stepsize * (x.grad * window_mask / grad_norms(x)**gamma).data.cpu().numpy()
        added_grad_accumulator = tuple([
            (grad_accumulator[idx][0] + grad(p_[0]),
             grad_accumulator[idx][1] + grad(p_[1]))
            for idx, p_ in enumerate(curr_perturbation)
        ])
        return added_grad_accumulator  
        
        
    def get_new_accumulated_hidden(self, model, curr_probs, curr_unpert_past, new_accumulated_hidden, wte):
        inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
        _, curr_unpert_past, curr_all_hidden = model(past_key_values=curr_unpert_past, inputs_embeds=inputs_embeds)[:]
        curr_hidden = curr_all_hidden[-1]
        new_accumulated_hidden = new_accumulated_hidden + torch.sum(curr_hidden, dim=1)
        return curr_unpert_past, new_accumulated_hidden    
        
        
    def get_kl_loss(self, kl_scale, unpert_probs_correct, probs, device):
        correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
        corrected_probs = probs + correction
        kl_loss = kl_scale * ((corrected_probs * (corrected_probs / unpert_probs_correct).log()).sum())
        return kl_loss
        
        

if __name__ == "__main__":
    from transformers.file_utils import cached_path
    
    set_seed(42)
    
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)

    class ClassificationHead(torch.nn.Module):
        def __init__(self, class_size, embed_size):
            super().__init__()
            self.class_size = class_size
            self.embed_size = embed_size
            self.mlp = torch.nn.Linear(embed_size, class_size)

        def forward(self, hidden_state):
            logits = self.mlp(hidden_state)
            return logits
    
    
    def get_classifier(class_size, embed_size, path, url=None):
        classifier = ClassificationHead(class_size=class_size,embed_size=embed_size)
        if url is not None:
            resolved_archive_file = cached_path(url)
            path = resolved_archive_file
        classifier.load_state_dict(torch.load(path, map_location='cpu'))
        classifier.eval()
        return classifier    

    # classifier = get_classifier(2,768, path='model_param/train_classifier_1/toxic_classifier_head_epoch_10.pt')
    classifier = get_classifier(2, 1024, path=None, url ="https://raw.githubusercontent.com/uber-research/PPLM/master/paper_code/discrim_models/toxicity_classifierhead.pt")


    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<pad>')
    # tokenizer.add_special_tokens({'additional_special_tokens': ['<sys>', '<usr>']})
    # model = GPT2LMHeadModel.from_pretrained('model_param/checkpoint-6600', output_hidden_states=True)
    # model.resize_token_embeddings(len(tokenizer))
    # model.eval()


    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium', output_hidden_states=True)
    model.eval()


    senten = "<|endoftext|>I don't know how to"
    # senten = "<usr>Hey dude, what's up!<|endoftext|><sys>Good!, How are you?<|endoftext|><usr>Nothing better! Anyway, how do you think about me?<|endoftext|><sys>"
    inputs = tokenizer(senten, return_tensors='pt')


    pgen = PPLMGeneration(model, classifier, device=device)

    import time
    start = time.time()
    gen = pgen.generate(inputs['input_ids'].to(device),
                max_length=20 + inputs['input_ids'].shape[-1]+1,
                min_length=20 + inputs['input_ids'].shape[-1],
                do_sample=True,
                temperature=1.0,
                top_k=10,
                repetition_penalty=1,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=None,
                num_return_sequences=10,
                perturb=True,
                class_label=1,
                horizon_length=1,
                window_length=0,
                decay=False,
                gamma=1.0,
                kl_scale=0.01,
                gm_scale=0.9,
                stepsize=0.07,
                num_iterations=10,
                grad_length=10000,
                )
    end = time.time()-start
    for idx, g in enumerate(gen):
        print(f'gen:{idx} ', [tokenizer.decode(g)])
    print(f'took {end:.6f} seconds for generation')