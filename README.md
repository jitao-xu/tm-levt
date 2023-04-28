# TM-LevT

Source code for the paper:

[**Integrating Translation Memories into Non-Autoregressive Machine Translation**](https://arxiv.org/abs/2210.06020)

Jitao Xu, Josep Crego, François Yvon

## Installation

Same as standard `fairseq`, please refer to the [installation instruction](https://github.com/jitao-xu/tm-levt/tree/main/fairseq-tm-levt#requirements-and-installation) in the `fairseq-tm-levt` directory.

## Data Preprocessing

The following command will preprocess parallel En-Fr data with augmented TM match for decoder input in `$data` for training with a shared vocabulary:

```
fairseq-preprocess \
    --source-lang en \
    --target-lang fr \
    --prev-target fr_prev \
    --trainpref $data/bpe.train.en-fr \
    --validpref $data/bpe.dev.en-fr \
    --destdir $data/bin \
    --joined-dictionary \
    --workers 40
```

## Training

Use this command to train a TM-LevT model with preprocessed data in `$data/bin`:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train \
    $data/bin \
    --task translation_lev \
    --criterion nat_loss \
    --arch levenshtein_transformer \
    --noise random_delete \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' \
    --warmup-updates 10000 \
    --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 \
    --share-all-embeddings \
    --dropout 0.3 \
    --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --save-dir models/$model \
    --tensorboard-logdir models/$model \
    --ddp-backend=legacy_ddp \
    --log-format 'simple' \
    --log-interval 200 \
    --fixed-validation-seed 7 \
    --max-tokens 8192 \
    --save-interval-updates 3000 \
    --keep-interval-updates 10 \
    --no-epoch-checkpoints \
    --max-update 300000 \
    --num-workers 2 \
    --skip-invalid-size-inputs-valid-test \
    --source-lang en \
    --target-lang fr \
    --fp16 \
    --prev-target fr_prev \
    --prev-target-prob 0.5 \
    --prev-del \
    --post-del \
    --prediction-for-prev-deletion
```

By default, we use the ``prev-target`` for training with a probability of 0.5 with the option ``--prev-target-prob``. The option ``--prev-del`` activates the initial deletion module. We use the union of initial deletion prediction and reference label with the option ``--prediction-for-prev-deletion``. The original LevT is equivalent to only using ``--post-del``.

## Inference

To generate translations with standard LevT decoding starting from scratcht into a seperated file as well as on `stdout`:

```
fairseq-interactive \
    $DATA_DIR \
    --path models/$CHECKPOINT_DIR/checkpoint_best.pt \
    --task translation_lev \
    --buffer-size 1024 \
    --source-lang en \
    --target-lang fr \
    --fp16 \
    --input $testset \
    --iter-decode-max-iter 10 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --print-step --retain-iter-history \
    --max-tokens 8192 \
| tee test_set.out
```

The `$DATA_DIR` directory contains the dictionary files for all languages. 

### Inference with TM match

To generate translations with the TM match as initialization on the target side:

```
fairseq-interactive \
    $DATA_DIR \
    --path models/$CHECKPOINT_DIR/checkpoint_best.pt \
    --task translation_lev \
    --buffer-size 1024 \
    --source-lang en \
    --target-lang fr \
    --fp16 \
    --input $testset \
    --iter-decode-max-iter 10 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --print-step --retain-iter-history \
    --prefix-size 1 \
    --max-tokens 8192 \
| tee test_set.out
```

## Acknowledgments

Our code was modified from [fairseq](https://github.com/pytorch/fairseq) codebase. We use the same license as fairseq(-py).

## Citation

```
@misc{xu2023integrating,
      title={Integrating Translation Memories into Non-Autoregressive Machine Translation}, 
      author={Jitao Xu and Josep Crego and François Yvon},
      year={2023},
      eprint={2210.06020},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
