"""Exercise the installed GRPO/HF/Accelerate scaling path on a tiny CPU model."""
import tempfile
import types
import unittest
import torch
import run_bfcl_intervention as r
from transformers import GPT2Config, GPT2LMHeadModel


class AccumulationContractTests(unittest.TestCase):
    def test_installed_path_retains_historical_accelerator_scaling(self):
        with tempfile.TemporaryDirectory() as out:
            tokenizer=r.AutoTokenizer.from_pretrained(r.ROOT/'smoke/model',local_files_only=True)
            model=GPT2LMHeadModel(GPT2Config(vocab_size=16,n_embd=8,n_layer=1,n_head=1))
            config=r.GRPOConfig(output_dir=out,use_cpu=True,bf16=False,per_device_train_batch_size=1,
                gradient_accumulation_steps=4,steps_per_generation=4,num_generations=4,beta=0.,
                report_to='none',disable_tqdm=True)
            trainer=r.BFCLTrainer(model=model,args=config,processing_class=tokenizer,
                train_dataset=r.Dataset.from_list([{'prompt':'x'}]*4),reward_funcs=lambda **kwargs:[0.]*4)
            self.assertEqual(trainer.accelerator.gradient_accumulation_steps,4)
            trainer.current_gradient_accumulation_steps=4
            trainer._prepare_inputs=types.MethodType(lambda self,inputs:inputs,trainer)
            param=next(model.parameters())
            # An analytic loss stands in for GRPO's already-normalized microbatch loss.
            trainer.compute_loss=types.MethodType(lambda self,model,inputs,**kw:param.sum()/4,trainer)
            trainer.training_step(model,{},None)
            self.assertTrue(torch.equal(param.grad,torch.full_like(param,1/16)))
            # A second microbatch accumulates, rather than replacing, the gradient.
            trainer.training_step(model,{},None)
            self.assertTrue(torch.equal(param.grad,torch.full_like(param,2/16)))


if __name__=='__main__':unittest.main()
