# AirDominace_1D_GAN_Git
Experiment to reproduce the results of 1D problem by using GAN in "Air Dominance Through Machine Learning", Rand Corporation, 2020

Japanese tutorials will be found at:
 - [ランド研究所の「機械学習による航空支配」を実装する（その1）：レポートのまとめ](https://qiita.com/DreamMakerAi/items/95e0aad16e450cb0c53d)
 - [ランド研究所の「機械学習による航空支配」を実装する（その2）：1次元問題について](https://qiita.com/DreamMakerAi/items/72c90f9df5339dedd9cd)
 - [ランド研究所の「機械学習による航空支配」を実装する（その3）： 1D simulator for GAN と Random mission planner の実装](https://qiita.com/DreamMakerAi/items/1a0e25a9f673e531ee6f)
 - [ランド研究所の「機械学習による航空支配」を実装する（その4）： conditional GAN の実装とトレーニング](https://qiita.com/DreamMakerAi/items/3a416d5cb64d9b7e86db)
 - [ランド研究所の「機械学習による航空支配」を実装する（その5）：トレーニング結果の分析](https://qiita.com/DreamMakerAi/items/6c78181b71c470632fcf)
 - [ランド研究所の「機械学習による航空支配」を実装する（その６）：トレーニング・データの重要性と GAN の性能向上](https://qiita.com/DreamMakerAi/items/b8ae6c9afd28e90b5b5c)
 
1. Use 'simple_1D_simulator_for_gan.py' to produce data for training, evaluation, and test.
2. Use 'train_conditional_gan.py' for training, evaluation, and test.
3. Use 'Transform Tensorboard data to pandas DataFrame all at once.ipynb' to get DataFrame from tensorboard.
4. Use 'Run GAM Mission Generator and check the performance.ipynb' to analyse the data.
