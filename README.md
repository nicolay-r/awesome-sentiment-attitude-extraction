# Awesome Sentiment Attitude Extraction

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of awesome studes related to sentiment attitude extraction,
in which **attitude** corresponds to the *text position* 
conveyed by <u>Subject</u> towards other <u>Object</u>
mentioned in text such as: entities, events, etc.
 
This repository collects works both related to **relation extraction** and **sentiment analysis** 
in which these two domains are inextricably linked, including event factualization as fundamentional studies 
for sentiment inference, stance detection.

Contributing: Please feel free to make *pull requests* or contact me [[contacts]](https://nicolay-r.github.io/)

### General and related studies
* [Natural Language Processing](https://github.com/keon/awesome-nlp#nlp-in-chinese)
    * [Sentiment Analysis](https://github.com/laugustyniak/awesome-sentiment-analysis)
        * [Targeted Setiment Analysis](https://arxiv.org/pdf/1905.03423.pdf)
        * [Aspect-based Sentiment Analysis](https://github.com/jiangqn/Aspect-Based-Sentiment-Analysis)
        * [Hate-speech detection](https://aclanthology.org/W17-1101.pdf)
    * [Relation Extraction](https://github.com/roomylee/awesome-relation-extraction)
    * Natural Language Inference

## Contents

* [Frameworks](#frameworks)
* [Papers](#papers)
    * [Language Models](#language-models)
    * [Neural-network based Models](#neural-network-based-models)
    * [Conventional Machine Learning Models](#conventional-machine-learning-models)
    * [CRF-based Models](#crf-based-models)
    * [Rule-based Verb-applicable Models](#rule-based-verb-applicable-models)
    * [Subsidiary Studies And Resources](#subsidiary-studies-and-resources)
    * [Miscellaneous](#miscellaneous)
* [Datasets](#datasets)

## Frameworks

* **AREkit** [[github]](https://github.com/nicolay-r/AREkit) [[research-applicable-paper]](https://arxiv.org/pdf/2006.13730.pdf)
    * Is an open-source and extensible toolkit focused on data preparation for 
    document-level relation extraction organization. 
     It complements the OpenNRE functionality, as in terms of the latter, 
     document-level RE setting is not widely explored (2.4 [[paper]](https://aclanthology.org/D19-3029.pdf)). 
* **DeRE**
 [[github]](https://github.com/ims-tcl/DeRE) 
 [[paper]](https://aclanthology.org/D18-2008/)
     * Is an open-source framework for **de**claritive **r**elation **e**xtraction, and therefore allows to declare your own task (using XML schemas) and apply manually implemented models towards it (using a provided API).
* **OpenNRE**
 [[github]](https://github.com/thunlp/OpenNRE) 
 [[paper]](https://aclanthology.org/D19-3029.pdf)
    * Is an open-source and extensible toolkit that provides a unified framework to implement neural models for relation extraction (RE) between named entities
* Others ... [[awesome-relation-extraction]](https://github.com/roomylee/awesome-relation-extraction/blob/master/README.md#frameworks)

## Papers

### Language Models
  * BERT-based models (Encoder Reprsentation From Transorfmers) 
  [[papers]](https://github.com/roomylee/awesome-relation-extraction#encoder-representation-from-transformer)
    * `Considering BERT model as classifier`
    * Joohong Lee, Awesome Relation Extraction  
  * GPT-based (Encoder Reprsentation From Transorfmers) 
  [[papers]](https://github.com/roomylee/awesome-relation-extraction#decoder-representation-from-transformer)
    * `Considering GPT model competed for classification task`
    * Joohong Lee, Awesome Relation Extraction  
    
[Back to Top](#contents)
	
### Neural-network based Models 
* No Permanent Friends or Enemies: Tracking Relationships between Nations from News
   [[paper]](https://arxiv.org/pdf/1904.08950)
    * Xiaochuang Han, Eunsol Choi, Chenhao Tan
    * NAACL-HLT 2019
* Neural networks for open domain targeted sentiment
    [[paper]](https://aclanthology.org/D15-1073.pdf)
    * Meishan Zhang, Yue Zhang, Duy-Tin Vo
    * ACL 2015
       
[Back to Top](#contents)

### Conventional Machine Learning Models

* Document-level Sentiment Inference with Social, Faction, and Discourse Context 
[[paper]](https://aclanthology.org/P16-1032.pdf)
	* Eunsol Choi, Hannah Rashkin, Luke Zettlemoyer, Yejin Choi
	* ACL-2016
	
### CRF-based Models
* Open Domain Targeted Sentiment 
    [[paper]](https://aclanthology.org/D13-1171.pdf)
    * Margaret Mitchell, Jacqueline Aguilar, Theresa Wilson, Benjamin Van Durme
    * ACL 2013

### Rule-based Verb-applicable Models
* Stance detection in Facebook posts of a German right-wing party 
    [[paper]](https://aclanthology.org/W17-0904.pdf)
    * Manfred Klenner, Don Tuggener, Simon Clematide
    * `Verb-usages form`
    * ACL 2017 (2nd Workshop on Linking Models of Lexical, Sentential and Discourse-level Semantics)
* An object-oriented model of role framing and attitude prediction 
    [[paper]](https://aclanthology.org/W17-6917.pdf)
    * `Object-oriented model`
    * ACL 2017 (2nd Workshop on Linking Models of Lexical, Sentential and Discourse-level Semantics)
* Joint Prediction for Entity/Event-Level Sentiment Analysis using Probabilistic Soft Logic Models 
    [[paper]](https://aclanthology.org/D15-1018.pdf)
    * Lingjia Deng, Janyce Wiebe
    * EMNLP 2015
* FactBank: a corpus annotated with event factuality 
    [[paper]](https://www.researchgate.net/profile/Roser-Sauri/publication/220147734_FactBank_A_corpus_annotated_with_event_factuality/links/0f31753144a2cdc1b5000000/FactBank-A-corpus-annotated-with-event-factuality.pdf)
    * Roser Saurí, James Pustejovsky  
    * 2009
       
[Back to Top](#contents)

### Subsidiary Studies and Resources
* Multilingual Connotation Frames: A Case Study on Social Media
for Targeted Sentiment Analysis and Forecast 
    [[paper]](https://aclanthology.org/P17-2073.pdf)
    [[resources]](https://hrashkin.github.io/multicf.html)
    * Hannah Rashkin, Eric Bell, Yejin Choi, Svitlana Volkova
    * ACL-2017
* Learning Lexico-Functional Patterns for First-Person Affect 
    [[paper]](https://aclanthology.org/P17-2022.pdf)
    * Lena Reed, Jiaqi Wu, Shereen Oraby
    * ACL-2017 
* Understanding Abuse: A Typology of Abusive Language Detection Subtasks
    [[paper]](https://aclanthology.org/W17-3012.pdf)
    * Zeerak Waseem, Thomas Davidson, Dana Warmsley, Ingmar Weber
    * ACL-2017
* Connotation Frames: A Data-Driven Investigation
    [[paper]](https://aclanthology.org/P16-1030.pdf)
    * Hannah Rashkin, Sameer Singh, Yejin Choi 
    * ACL-2016
* Do Characters Abuse More Than Words?
    [[paper]](https://aclanthology.org/W16-3638.pdf)
    * Yashar Mehdad, Joel Tetreault
    * SIGDIAL-2016
   
[Back to Top](#contents)

### Miscellaneous

* Verifying the robustness of opinion inference [[paper]](https://core.ac.uk/reader/83654780)
    * Josef Ruppenhofer, Jasper Brandes
    * KONVENS 2016
   
[Back to Top](#contents)

## Datasets

* NOW (2010 -- present) 
  [[site]](https://www.corpusdata.org/now_corpus.asp) --
    News on the Web Corpus.
    * Contains data from online magazines and newspapers in 20 different English-speaking countries from 2010 to the current time.
    (Raw texts only).
* MPQA-3.0, (2015)
    [[site]](https://mpqa.cs.pitt.edu/) 
    [[paper]](https://aclanthology.org/N15-1146.pdf)
* SNLI 
    [[site]](https://nlp.stanford.edu/projects/snli/) 
    [[paper]](https://nlp.stanford.edu/pubs/snli_paper.pdf)  -- 
    Stanford Natural Language Inference 
    * 570k human-written English sentence pairs manually labeled for balanced classification with the labels 
    *entailment*, *contradiction*, and *neutral*
* FactBank 2009, 
    [[paper]](https://www.researchgate.net/profile/Roser-Sauri/publication/220147734_FactBank_A_corpus_annotated_with_event_factuality/links/0f31753144a2cdc1b5000000/FactBank-A-corpus-annotated-with-event-factuality.pdf) --
    a corpus annotated with event factuality
    * Consists of 208 documents and contains a total of 9,488, including TimeBank data;
manually annotated events.
* TimeBank, 2003
    [[site]](http://www.timeml.org/site/timebank/timebank.html)
    [[paper]](https://www.researchgate.net/profile/James-Pustejovsky/publication/228559081_The_TimeBank_corpus/links/09e4150ca6331b2eb9000000/The-TimeBank-corpus.pdf)
    * Annotated to indicate events, times, and temporal relations
   
[Back to Top](#contents)
