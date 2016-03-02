## このプログラムについて

最近話題の LSTM (Long-Short Term Memory) を使って実験的に作った形態素解析エンジンです．もともとは，[Chainer](http://chainer.org/) の練習用に書いてみたもので，公開する予定もなかったのですが，思いのほか高い性能が得られたので公開することにしました．

分かち書きだけでなく，形態素解析も実装しているのですが，まだデータが整理できていないので，ドキュメントは準備中です．

## 使い方

### 分かち書き
#### 入力データ
単語を半角スペースで区切り，各行に1文ずつ並べたテキストファイルを入力として受け取ります．

例：
```
これ は テスト です
今日 は いい 天気 です
```

#### 訓練
```python
python morpheme/segmentation.py train path/to/train.txt path/to/output/directory
```

#### テスト
```python
python morpheme/segmentation.py test path/to/train.txt path/to/output/directory epoch
```

### 形態素解析
準備中

## 工夫点

### 辞書を使わない

出現する文字を事前に辞書化せずに，byte 単位で処理しています．実際のところ．1000文程度学習すれば，文字コードの途中で分割するようなことは，まず起こりません．これによって，辞書に載っていない文字が現れた時に処理できないという問題が発生せず，UTF-8で表現された任意の文章を扱うことができます．

### ネットワークの改良

[chainer_examples](https://github.com/odashi/chainer_examples) や [Bi-directional LSTM Recurrent Neural Network for Chinese Word Segmentation](http://arxiv.org/abs/1602.04874) では，

## 実験

現状，ちゃんと計測したのは，単語分割の F値 だけです．
品詞推定，活用形，活用型に関してはざっと見た感じ，0.80〜0.90 ぐらいだと思います．

### その1
[@odashi_t さんの書いた Qiita の記事](http://qiita.com/odashi_t/items/a1be7c4964fbea6a116e) 
に載っているデータセットを使って，単語分割オンリーで学習させてみました．

データには `train50000.ja` の先頭から10000文，テストには `test1000.ja` を使っています．

このデータセットは，比較的クセのない日本語なので，
10000文学習すれば 98% の精度で分割できます．

```text
1000
precision: 0.950255326642
recall: 0.954879235601
F-measure: 0.952561669329

5000
precision: 0.967938665273
recall: 0.982924887198
F-measure: 0.975374214855

10000
precision: 0.983007345783
recall: 0.982659470937
F-measure: 0.982833377077
```

一方で，[chainer_examples](https://github.com/odashi/chainer_examples) 版は以下のような精度になります．

```text
1000
precision: 0.799856585133
recall: 0.974376395225
F-measure: 0.878533298329

5000
precision: 0.863014899211
recall: 0.955741046297
F-measure: 0.907014231106

10000
precision: 0.896908159532
recall: 0.951664563719
F-measure: 0.923475394396

20000
0.91850117096 0.951664563719 0.93478882639

30000
0.926962585996 0.954673396098 0.940613942813

40000
0.931468002647 0.95642046006 0.943779331482

50000
0.93455645924 0.959138115112 0.946687742492
```
