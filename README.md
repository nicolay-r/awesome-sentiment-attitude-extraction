# Awesome Sentiment Attitude Extraction

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<p align="center">
    <img src="logo.png"/>
</p>

A curated list of awesome studes related to sentiment attitude extraction,
in which **attitude** corresponds to the *text position* 
conveyed by <u>Subject</u> towards other <u>Object</u>
mentioned in text such as: entities, events, etc.
 
This repository collects works both related to **relation extraction** and **sentiment analysis** 
in which these two domains are inextricably linked, including event factualization as fundamentional studies 
for sentiment inference, stance detection.

Contributing: Please feel free to make *pull requests* or contact me [[contacts]](https://nicolay-r.github.io/)

## Contents

* [Related Studies](#related-studies)
* [Frameworks](#frameworks)
* [Annotation Schemas](#annotation-schemas)
* [Papers](#papers)
    * [Large Language Models](#large-language-models)
      * [Fact-Checking Adaptation](#fact-checking-adaptation)
      * [Chain-of-Thought](#chain-of-thought)
      * [Conversational Systems](#conversational-systems)
    * [Language Models](#language-models)
    	* [Graph-Based](#graph-based)
    	* [Low Resource Tunings](#low-resource-tunings)
    	* [Prompts and Knowledge Examination](#prompts-and-knowledge-examination)
        * [Architectures](#architectures) 
    * [Conventional neural-network based Models](#conventional-neural-network-based-models)
    * [Conventional Machine Learning Models](#conventional-machine-learning-models)
    * [CRF-based Models](#crf-based-models)
    * [Rule-based Verb-applicable Models](#rule-based-verb-applicable-models)
    * [Subsidiary Studies And Resources](#subsidiary-studies-and-resources)
    * [Miscellaneous](#miscellaneous)
* [Thesises](#thesises)
* [Datasets](#datasets)

## Related studies
* [Natural Language Processing](https://github.com/keon/awesome-nlp#nlp-in-chinese)
    * [Sentiment Analysis](https://github.com/laugustyniak/awesome-sentiment-analysis)
        * [Targeted Setiment Analysis](https://arxiv.org/pdf/1905.03423.pdf)
        * [Structured Sentiment Analysis](https://aclanthology.org/2022.semeval-1.180.pdf) (SemEval Task 10)
        * [Aspect-based Sentiment Analysis](https://github.com/jiangqn/Aspect-Based-Sentiment-Analysis)
        * [Hate-speech detection](https://aclanthology.org/W17-1101.pdf)
    * [Relation Extraction](https://github.com/roomylee/awesome-relation-extraction)
    * [Stance Detection](https://github.com/sumeetkr/AwesomeStanceLearning)
  
## Frameworks
* **bulk-chain** [[github]](https://github.com/nicolay-r/bulk-chain) 
    * Framework that exploits Chain-of-Thought concept and provides minimalistic solution for zero-shot inferences. For example, you can exploit the concept of `aspect-opininon-reason` chain from  [THOR-ISA](https://github.com/scofield7419/THOR-ISA) to adapt it for attitude extraction.
* **FaiMA** [[github]](https://github.com/SupritYoung/FaiMA)
    * Framework that integrates graph-based models and linguistics, with a core feature aimed at in-context learning for multi-domain SA.
* **Reasoning-for-Sentiment-Analysis-Framework** [[github]](https://github.com/nicolay-r/Reasoning-for-Sentiment-Analysis-Framework)
    * This frameworks repesent a reforget 🛠️ version of the `THOR-ISA` framework: 
    * **THOR-ISA** [[github]](https://github.com/scofield7419/THOR-ISA)
        * Propt-based framework for setiment Analysis that based on Chain-of-Though concept for obtaining the result sentiment class out of the LLM system.	
* **OpenPrompt** [[github]](https://github.com/thunlp/OpenPrompt)
    * Enhanced tool for automatic completion of the prompt via the provided resources.	
* **ChatGPT** [[site]](https://openai.com/blog/chatgpt/)
    * Conversation system that is trained to follow the instruction in a prompt and provide a detailed response; 
      examples on how it could be adapted reviewed in the following [work](https://arxiv.org/pdf/2212.14548.pdf).
* **arekit-prompt-sampler** 
  [[github]](https://github.com/nicolay-r/arekit-prompt-sampler) 
  [[prompt-engeneering-guide]](https://github.com/dair-ai/Prompt-Engineering-Guide)
    * Sentiment Attitude Extraction sources sampling with language 
    transferring and prompting API for further ChatGPT-alike model requests, powered by [AREkit](https://github.com/nicolay-r/AREkit).
* **ARElight** [[github]](https://github.com/nicolay-r/ARElight)
    * [AREkit-based](https://github.com/nicolay-r/AREkit) application for a granular view onto sentiments between entities in a mass-media texts written in Russian
* **AREnets** [[github]](https://github.com/nicolay-r/AREnets)
    * Is an OpenNRE like project, but the kernel based on tensorflow library, with implementation of neural networks on top of it, designed for Attitude and Relation Extraction tasks.
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
    * Is an open-source and extensible toolkit that provides a unified framework to implement neural models for relation extraction (RE) between named entities.
* **DeepPavlov-0.17.0** 
[[docs]](http://docs.deeppavlov.ai/en/0.17.0/features/models/re.html) 
[[post]](https://medium.com/deeppavlov/relation-extraction-for-deeppavlov-library-d1f7b57365b3)
    * Is an entire relation extraction component for DeepPavlov opensource library, proposed by Anastasiia Sedova.
* Others ... [[awesome-relation-extraction]](https://github.com/roomylee/awesome-relation-extraction/blob/master/README.md#frameworks)

[Back to Top](#contents)

## Annotation Schemas
* **OpinionML** [[paper]](https://www.researchgate.net/publication/332423185_OpinionML-Opinion_Markup_Language_for_Sentiment_Representation)
* **SentiML** [[paper]](https://dl.acm.org/doi/10.1145/2517978.2517994)
* **OpinionMiningML** [[paper]](https://d1wqtxts1xzle7.cloudfront.net/47692116/OpinionMining-ML20160801-28120-mzgsge-libre.pdf?1470049517=&response-content-disposition=inline%3B+filename%3DOpinionMining_ML.pdf&Expires=1667567660&Signature=L~lOd1CoiQGRU8X28xfKiEJbXXThItxUEpOx9uSS62nUhP9MBaR-1-XCVnKk1brFLUq5X1ooMkj0MCdGdnEPHwl7mLJLFmMbko9od207~EYvsbPyvPl9N6R9ceQMj3wH-W2A6EEigBZ8hTPxbAV6HWPOgFzIPOlyBS20-0o6SMTdtEFny714EtoVfS-E941qliBJyHdcOYVzT-uf4MHrceBHhKvfpwe0xDdLDC4QLVbbYbfDuWgbak1QEm7RKwQEITGeYE8zK5~1YIJT~MPvlP7aSbyPOjAfMpXbh2QCkBJC2KSY9q19pQOQz4uGtWsXQFbSRSLFxDFCK00ynuBccw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
* **EmotionML** [[paper]](https://www.researchgate.net/publication/221622141_EmotionML_-_An_Upcoming_Standard_for_Representing_Emotions_and_Related_States)

[Back to Top](#contents)

## Papers

	
[Back to Top](#contents)

### Large Language Models
> [Awesome-LLM list](https://github.com/Hannibal046/Awesome-LLM)

#### Fact-Checking Adaptation

* Consistent Document-Level Relation Extraction via Counterfactuals [[paper]](https://www.arxiv.org/abs/2407.06699) [[code]](https://github.com/amodaresi/CovEReD)
    * `The key concept is to use factual relations for fictional context construction and LLM validation`
    * Ali Modarressi, Abdullatif Köksal, Hinrich Schütze
    * **EMNLP-2024, 15th of October 2024**

#### Chain-of-Thought
* [FaiMA](https://github.com/SupritYoung/FaiMA): Feature-aware In-context Learning for Multi-domain
Aspect-based Sentiment Analysis 
[[paper]](https://arxiv.org/pdf/2403.01063.pdf)
[[code]](https://github.com/SupritYoung/FaiMA)
    * [Framework](https://github.com/SupritYoung/FaiMA) `that integrates` [graph-based models](#graph-based)` and lingustics, with core feature aimed at in-context-learning feature for multi-domain SA; The framework is designed for multidomain datasets; 
      Due to graphs and pairs-generation module, it may find major contribution in **attitude-based** sentiment extraction and target-oriented SA.`
    * Songhua Yang, Xinke Jiang, Hanjie Zhao, Wenxuan Zeng, Hongde Liu, Yuxiang Jia
    * **LREC-COLING 2024, Long Paper**; Submitted 2 Mar. 2024.
* Aspect-Based Sentiment Analysis with Explicit Sentiment Augmentations
  [[paper]](https://arxiv.org/abs/2312.10961)
  [[harvard-paper]](https://ui.adsabs.harvard.edu/abs/2023arXiv231210961O/abstract)
    * `integrates explicit sentiment augmentations, acted as <<clues>> that augment LLM input context`
    * Jihong Ouyang, Zhiyao Yang, Silong Liang, Bing Wang, Yimeng Wang, Ximing Li
    * Arxiv Pre-print, submitted: 18 Dec. 2024
* Sentiment Analysis through LLM Negotiations 
 [[paper]](https://arxiv.org/abs/2311.01876) 
 [[open-review]](https://openreview.net/pdf?id=1VlIXyAw04k)
    * `generator-discriminator of negotiating the result label`
    * Xiaofei Sun, Xiaoya Li, Shengyu Zhang, Shuhe Wang, Fei Wu, Jiwei Li, Tianwei Zhang, Guoyin Wang
    * Arxiv Pre-print, submitted: 2024
* Reasoning Implicit Sentiment with Chain-of-Thought Prompting [[paper]](https://aclanthology.org/2023.acl-short.101.pdf) [[code]](https://github.com/scofield7419/THOR-ISA)
	* `Sequence of 3 prompts for conversational system, complemented by tge system responses. Reason is to cope with hallucination` [similar-studies](https://openreview.net/pdf?id=1PL1NIMMrw)  	
	* Hao Fei, Bobo Li, Qian Liu, Lidong Bing, Fei Li, Tat-Seng Chua
	* **ACL 2023, Short Papers**

#### Conversational Systems
  > Using [Language Models](#language-models) (usually LARGE-sized) in a combination with promts/questions 
   * Sentiment Analysis in the Era of Large Language Models: A Reality Check
    [[paper]](https://arxiv.org/pdf/2305.15005.pdf)
        * `application of the LLM and based on the latter ChatGPT for the variety set of sentiment analysis problems`
        * Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Jialin Pan, Lidong Bing 
        * arXiv, 24 May 2023
  * Is ChatGPT better than Human Annotators? Potential and Limitations of ChatGPT in Explaining Implicit Hate Speech 
  [[paper]](https://arxiv.org/pdf/2302.07736.pdf)
	* Huang Fan, Kwak Haewoon, An Jisun
	* Harvard, Februrary, 2023
  * How would Stance Detection Techniques Evolve after the Launch of ChatGPT?
  [[paper]](https://arxiv.org/pdf/2212.14548.pdf)
  	* `Introducing prompt templater which allows to reach state-of-the art with zero-shot learning!`
  	* Bowen Zhang, Daijun Ding, Liwen Jing
	* Harvard, December, 2022

### Language Models
> [Awesome-LLM list](https://github.com/Hannibal046/Awesome-LLM)

#### Graph-Based

  * Comparing `Graph-` and `Seq2Seq-` based Models Highlights Difficulty in Structured Sentiment Analysis
  [[paper]](https://aclanthology.org/2022.semeval-1.188.pdf)
  [[code]](https://github.com/hitachi-nlp/graph_parser)
  	* Gaku Morio, Hiroaki Ozaki, Atsuki Yamaguchi, and Yasuhiro Sogawa
	* ACL-Workshop, 2022
  * Enhancing Zero-shot and Few-shot Stance Detection with Commonsense Knowledge Graph
  [[paper]](https://aclanthology.org/2021.findings-acl.278.pdf)
  	* Rui Liu, Zheng Lin, Yutong Tan1, Weiping Wang
	* ACL-IJCNLP 2021
	
[Back to Top](#contents)

#### Low Resource Tunings

  * Zero-shot Sentiment Analysis in Low-Resource Languages Using a Multilingual Sentiment Lexicon
    [[paper]](https://arxiv.org/pdf/2402.02113.pdf)
    [[code]](https://github.com/fajri91/ZeroShotMultilingualSentiment)
    * Fajri Koto, Tilman Beck, Zeerak Talat, Iryna Gurevych, Timothy Baldwin
    * NAACL-2024
  * Black-Box Tuning for Language-Model-as-a-Service
    [[paper]](https://arxiv.org/pdf/2201.03514.pdf)
    [[code]](https://github.com/txsun1997/Black-Box-Tuning)
    * `Non gradient p-tunes, wrapped in API in order to consider large Pre-Trained models (PTMs) adoptation as Service models`
    * Tianxiang Sun, Yunfan Shao, Hong Qian, Xuanjing Huang, Xipeng Qiu
    * Arxiv Pre-print, 2022
  * P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks 
  	[[paper]](https://arxiv.org/pdf/2110.07602.pdf)
	[[code]](https://github.com/THUDM/P-tuning-v2)
  	* `Proceeds Prefix-Tuning idea onto multiple layers of LM-model`
	* Xiao Liu, Kaixuan Ji, Yicheng Fu, Zhengxiao Du, Zhilin Yang, Jie Tang
	* Dblp Jornal, 2021
  * The Power of Scale for Parameter-Efficient Prompt Tuning 
	[[paper]](https://aclanthology.org/2021.emnlp-main.243.pdf)
	[[code]](https://github.com/google-research/prompt-tuning)
	* `Prompt-designing, prompt-tuning comparison studies`
	* Brian Lester, Rami Al-Rfou, Noah Constant
	* EMNLP-2021
  * GPT Understands, Too 
        [[paper]](https://arxiv.org/pdf/2103.10385.pdf) 
        [[code]](https://github.com/THUDM/P-tuning)
	* `Promt Tuning (p-tuning), i.e. training only promt token embeddings before and after input sequence (x)`
	* Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, Jie Tang
	* 2021
  * Prefix-Tuning: Optimizing Continuous Prompts for Generation 
	[[paper]](https://aclanthology.org/2021.acl-long.353.pdf)
	[[code]](https://github.com/XiangLi1999/PrefixTuning)
	* `Training token prefixes for downstream tasks with frozen LM parameters` 
	* Xiang Lisa Li, Percy Liang
	* ACL/IJCNLP-2021
  * Language Models are Few-Shot Learners 
	[[paper]](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
	* `Prompt designing. FS, 1S by presenting context as "[input,result] x k-times", where k > 1 (FewShot), k = 1 (OneShot); ZeroShot includes only descriptor of expected result`
	* Tom B. Brown, et. al.
	* NeurIPS-2020
  * AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts 
   	[[paper]](https://aclanthology.org/2020.emnlp-main.346.pdf)
	[[code]](https://github.com/ucinlp/autoprompt)
	* `Considering sentiment analysis task as MLM by predicting [MASK]; prompting input (x) with tokens (p1...pk), selected by gradient search (considering that label has corresponding tokens (prompts))`
	* Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, Sameer Singh
	* EMNLP-2020

[Back to Top](#contents)

#### Prompts and Knowledge Examination

  * Sentiment Analysis in the Era of Large Language Models: A Reality Check
    [[paper]](https://arxiv.org/pdf/2305.15005.pdf)
    * `duplicated from the one in conversational systems section`
    * Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Jialin Pan, Lidong Bing
    * arXiv, 24 May 2023
  * How Can We Know What Language Models Know? 
    [[paper]](https://aclanthology.org/2020.tacl-1.28.pdf) 
    [[code]](https://github.com/jzbjyb/LPAQA)
	* `Implemented model LPAQA: Language model Prompt And Query Archive`
	* Zhengbao Jiang, Frank F. Xu, Jun Araki, Graham Neubig
	* TACL-2020
  * Language Models as Knowledge Bases? 
    [[paper]](https://aclanthology.org/D19-1250.pdf)
    [[code]](https://github.com/facebookresearch/LAMA)
  	* Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, Sebastian Riedel
	* EMNLP-2019	
  * Utilizing BERT for Aspect-Based Sentiment Analysis
via Constructing Auxiliary Sentence 
    [[paper]](https://aclanthology.org/N19-1035.pdf) 
    [[code]](https://github.com/HSLCY/ABSA-BERT-pair)
	* `Adopting a predefined prompt (QA/NLI formats) as a TextB input part`
	* Chi Sun, Luyao Huang, Xipeng Qiu
	* NAACL-HLT 2019

[Back to Top](#contents)

#### Architectures
  * BERT-based models (Encoder Reprsentation From Transorfmers) 
  [[papers]](https://github.com/roomylee/awesome-relation-extraction#encoder-representation-from-transformer)
    * `Considering BERT model as classifier`
    * Joohong Lee, Awesome Relation Extraction  
  * GPT-based (Encoder Reprsentation From Transorfmers) 
  [[papers]](https://github.com/roomylee/awesome-relation-extraction#decoder-representation-from-transformer)
    * `Considering GPT model competed for classification task`
    * Joohong Lee, Awesome Relation Extraction  
  * Comparing `Graph-` and `Seq2Seq-` based Models Highlights Difficulty in Structured Sentiment Analysis
  [[paper]](https://aclanthology.org/2022.semeval-1.188.pdf)
  [[code]](https://github.com/hitachi-nlp/graph_parser)
  	* `T5 and mT5 finetunnning`, i.e. 
	[Text-To-Text Transfer Transoformer](https://github.com/google-research/text-to-text-transfer-transformer) application
  	* Gaku Morio, Hiroaki Ozaki, Atsuki Yamaguchi, and Yasuhiro Sogawa
	* ACL-Workshop, 2022
    
[Back to Top](#contents)

### Conventional Neural-network based Models 

In this section we consider neural-network models based on convolutional, recurrent, recursive architectures.

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
* Sentiment Analysis: Capturing Favorability Using Natural Language Processing [[paper]](https://dl.acm.org/doi/pdf/10.1145/945645.945658)
	* `it is originally called favorability analysis, semantic establishment between sentiment and subject`
	* Tetsuya Nasukawa, Jeonghee Yi 
	* K-CAP-2003 (ACM)
	
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
* RIVETER Measuring Power and Social Dynamics Between Entities
  [[paper]](https://aclanthology.org/2023.acl-demo.36.pdf)
    * Maria Antoniak, Anjalie Field, Jimin Mun, Melanie Walsh, Lauren F. Klein, Maarten Sap
    * ACL-2023
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

## Thesises

* Mitigation of Gender Bias in Text using Unsupervised Controllable Rewriting [[master-thesis]](https://en.cs.uni-paderborn.de/fileadmin/informatik/fg/css/teaching/theses/brinkmann21-ma-thesis.pdf)
    * Maja Brinkmann
    * Paderborn University, 2022
    	* `Connotation Frames` (2.1.3.) 
    	* `Connotational Frames and Lexicon` (3.1.1.)

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
