"""Candidate explicit single GRPO normalization; not enabled in archived runners."""


def configure_single_normalization(trainer):
    args=trainer.args
    if args.gradient_accumulation_steps!=4 or args.per_device_train_batch_size!=1:
        raise ValueError('Only the accepted four-microbatch configuration is targeted')
    if args.num_generations!=4 or args.num_iterations!=1 or args.loss_type!='grpo':
        raise ValueError('Different objectives require separate normalization acceptance')
    if trainer.is_deepspeed_enabled or trainer.is_fsdp_enabled or trainer.accelerator.num_processes!=1:
        raise ValueError('Single-process ordinary optimizer only')
    if trainer.accelerator.gradient_accumulation_steps!=4:
        raise ValueError('Unexpected inherited accelerator normalization')
    # HF performs accumulation and synchronization explicitly; GRPO already divides by four.
    # Do not change HF batch counts, generation grouping, seeds, or the GRPO loss itself.
    trainer.accelerator.gradient_accumulation_steps=1
    assert trainer.args.gradient_accumulation_steps==4
    return {'hf_microbatches_per_update':4,'grpo_loss_divisor':4,
            'accelerate_backward_divisor_before':4,'accelerate_backward_divisor_after':1,
            'scope':'Candidate single-process GRPO normalization, CPU objective-gradient equivalence only.'}
