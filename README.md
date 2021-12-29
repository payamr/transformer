Transformer implementation adapted from the [tensorflow example](https://www.tensorflow.org/text/tutorials/transformer). Models trained on the [Portuguese to English](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en) ted datasets, the same as the tensorflow example. 

A small and large model are trained, as detailed by `configs` files. For both, `val_accuracy` reaches a roughly `0.35` maximum. `BLEU` and `GLEU` scores are calculated on the validation set after training using a few saved checkpoints. See the `inference.ipynb` notebook for results. 