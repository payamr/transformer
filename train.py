from model import Transformer
from datasets import translation_train_and_val_from_tfds, pt_to_en_tokenizers
from loss import masked_cross_entropy
from metrics import masked_accuracy, bleu_score, gleu_score
from optimizer import *
import yaml
import os
from typing import Dict
import argparse


MODEL_TO_TOKENIZER_FN = {
    "ted_hrlr_translate_pt_en_converter": pt_to_en_tokenizers
}

DISTRIBUTE_STRATEGIES = {
    'mirrored': tf.distribute.MirroredStrategy,
    'multiworker': tf.distribute.MultiWorkerMirroredStrategy,
}


def training_datasets(config: Dict, cache_dir: str) -> Dict:
    dataset_params = config['dataset_params']
    input_tokenizer, target_tokenizer = MODEL_TO_TOKENIZER_FN[
        dataset_params['translate_tokenizers_model']
    ](cache_dir)
    tf_dataset_name = dataset_params['tf_dataset_name']

    train_ds, val_ds = translation_train_and_val_from_tfds(
        tf_dataset_name,
        batch_size=dataset_params['batch_size'],
        tokenizer_input=input_tokenizer,
        tokenizer_target=target_tokenizer,
    )

    return {
        'train_ds': train_ds,
        'val_ds': val_ds,
        'input_vocab_size': input_tokenizer.get_vocab_size().numpy(),
        'target_vocab_size': target_tokenizer.get_vocab_size().numpy(),
        'input_tokenizer': input_tokenizer,
        'target_tokenizer': target_tokenizer,
    }


def build_model(config: Dict, data: Dict) -> tf.keras.Model:
    model_params = config['model_params']

    transformer = Transformer(
        num_layers=model_params['num_layers'],
        d_model=model_params['d_model'],
        num_heads=model_params['num_heads'],
        dff=model_params['dff'],
        input_vocab_size=data['input_vocab_size'],
        target_vocab_size=data['target_vocab_size'],
        max_posit_encode_input=model_params['max_posit_encode_input'],
        max_posit_encode_target=model_params['max_posit_encode_target'],
        dropout_rate=model_params['dropout_rate']
    )
    return transformer


def compile_model(config: Dict, data: Dict, model: tf.keras.Model) -> None:

    eager = False
    metric_params = config['metric_params']
    metrics = [masked_accuracy]
    if metric_params is not None:
        bleu = metric_params.get('bleu')
        if bleu:
            eager = True
            for i in range(1, bleu['max_order'] + 1):
                metrics.append(bleu_score(target_tokenizer=data['target_tokenizer'], order=i))
        if 'gleu' in metric_params:
            eager = True
            metrics.append(gleu_score(target_tokenizer=data['target_tokenizer']))

    learning_rate = CustomSchedule(config['model_params']['d_model'])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    model.compile(
        optimizer=optimizer,
        loss=masked_cross_entropy,
        metrics=metrics,
        run_eagerly=eager,
    )


def train_model(config: Dict, data: Dict, arguments: argparse.Namespace, model: tf.keras.Model):
    train_params = config['train_params']

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(arguments.checkpoint_dir, 'weights.{epoch:02d}.hdf5'),
            save_weights_only=True,
            verbose=1,
            save_freq='epoch' if train_params.get('save_freq') is None else int(train_params['save_freq'])
        ),
        tf.keras.callbacks.TensorBoard(log_dir=arguments.checkpoint_dir)
    ]

    model.fit(
        data['train_ds'],
        validation_data=data['val_ds'],
        epochs=train_params['epochs'],
        steps_per_epoch=train_params['steps_per_epoch'],
        validation_steps=train_params['validation_steps'],
        callbacks=callbacks,
    )


def build_and_train_model(config: Dict, data: Dict, arguments: argparse.Namespace):
    model = build_model(config, data)
    compile_model(config, data, model)
    train_model(config, data, arguments, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, default='config_large')
    parser.add_argument("--cache-dir", '-d', type=str, default='/home/payam/Documents/translate_data')
    parser.add_argument("--checkpoint-dir", '-p', type=str, default='/home/payam/Documents/checkpoints/transformer_pt_en_large')
    parser.add_argument("--distribute", '-s', type=str, default='mirrored', help="'mirrored', '', 'multiworker'")

    args = parser.parse_args()

    cfg_filename = f'{args.config}.yml'
    with open(os.path.join('configs', cfg_filename), 'r') as fym:
        cfg = yaml.load(fym)

    datasets = training_datasets(cfg, args.cache_dir)

    if args.distribute:
        strategy = DISTRIBUTE_STRATEGIES[args.distribute]()
        with strategy.scope():
            build_and_train_model(cfg, datasets, args)
    else:
        build_and_train_model(cfg, datasets, args)



