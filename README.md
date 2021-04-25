# Sentiment Analysis of Twitter tweets

* [data analysis](analysis.ipynb) 
* [preprocessing functions](./src/preprocessing.py)
* [inference script](inference.ipynb)

## The Process

### The objective

The purpose of this exercise was to determine the overall sentiment of 100k tweets, which is equivalent to assigning the individual tweets to one of the predefined categories denoting the polarity of a message. This task can be seen as either binary classification problem (two polarity categories: positive or negative) or multiclass classification problem (e.g. positive, neutral, negative). In the case of tweets, where some of the posts are meant to just convey the information, the later option seems to be the more suitable.

### The approach

Since none of the received tweets has been labeled with a ground truth label, there are two approaches that I came up with:
* Approach 1: use a pretrained model, trained on the similar task, to classify the received tweets
* Approach 1: manually label the subset of tweets and use them as training/validation sets to train either completly new model or apply transfer learning 

Becuse of the time constraints and most likely worse performance of the second approach, I've chosen the first option.

### The data

After a small research, I've started to investigate the received data.

#### Data quality issue

First thing that I've noticed is that some of the tweets appears to be corrupted. For instance, the first tweet in the dataset (in both a local file and on the Google Drive) is as follows:

> In case you don’t speak Bankese, we’ve translated it for you.
> Because knowing how much it really costs to send money abr…

while the original one looks like this:

![Corrupted example](/img/corrupted.png)

It turned out that the length of this tweet is exactly equal to 140 characters, which is a old twitter's limit and also sign of a larger problem with the dataset. I've checked the distribution of the tweets' length and it appears that large chunk of the dataset is distorted:

![Length distribution](/img/length-dist.png)

Given more time for a task I would probably use Twitter API to re-download the 140 characters tweets.

#### Data analysis

Below please find the summary of the analysis:

* Some tweets are significantly larger than twitter's 280 character limit - it turns out that names that auto-populate at the start of a reply tweet don't count towards the length, e.g.:
>@GillesnFio @DrSusanBarring1 @ATomalty @HenryK_B_ @Willard1951 @NoelTurner194 @yankeepirate247 @walther_jeff @EuphoricEuler @dan613 @RoyPentland @saminhim @elqulime @DawnTJ90 @NikolovScience @jimdtweet @moaningbs @uk_ecology @AtomsksSanakan @BakkMatt @MexicoRS78 @PierreTherrie14 @brjma @RetributiaNorb @PeletteSean @EastWhately @Kenneth72712993 @HMS_Indomitable @PolitiPeriphery @Michael46830937 @MedBennett @DaveOx13 @Eradinotte @PeterJrgen12 @jch_of @JSegor @Tony__Heller @d_e_mol @rln_nelson @Chrisdebois1 @MassiMassian @banurfeels @AngstromU @waitwha35825253 @dhaessel0 @RadioFreeTony @MonkeyMyBack @NeilSul70388398 @RushhourP @mtnman0038 Classic climate change denial.\n\nThey say 'look how bad green solutions are' forgetting the millions that die due to fossil fuel created air pollution &amp; ground water contamination. The oil wars etc. And of course let's not forget #ClimateChange \n\nhttps://t.co/EikyPTZli4

* All tweets in the dataset are assigned with the English language label
* All tweets are unique as far as id is concerned (these are unique posts) but there are duplicated as far as the content of a tweets is concerned. The duplicated are caused by retweets and short messages

* The retweets start with "RT @user" character string. I've found one case where retweet started with "RT Wikifactory"
* Some special characters like "\n" are present in the data and need to be removed
* The tweets contain a lot of acronyms (e.g. lol), emoticons (e.g. :D), terms expressing emotions (e.g. Woah), hyperlinks to external websites, other users' identifiers (e.g. @producerknoidea) and hashtags (e.g. #coronavirus)
* Spelling mistakes and the sequences of repeating characters (e.g. cooool insted of cool) could potentially be present in the data
* There are also quotes that can influence the sentiment assigned by a model to tweets. For instance the quote can have the positive sentiment while the overall tweet is rather neutral:
>Jan explains it’s something “which is very joyful, very grounding, and for writers to have a child in their lives is very very nice”

All things considered, there is a lot of noise in the data that can potentially hinder the inference.

### Text preprocessing

As a result of data analysis, I've decided to clean the dataset. Taking into account the research that I've done, the intuition and testing the model's performance (that I knew that I will use) on iteratively preprocessed text, I applied the following data transformationg:

* Removing all retweet information ("RT @user" ==> "")
* Removing urls
* Removing hashtags in front of the words ("#Tesla" ==> "Tesla")
* Removing other users' identifiers (targets)
* Removing new line characters
* Replacing abbreviations using constructed dictionary (I've found two already contructed dictionaries and combined them)
* Replacing three+ consecutive characters with two (e.g. "coool" ==> "cool")

Punctuations, emoticons, stop words, lemmatization/stemming were not removed/replaced/applied because the selected model can handle the text as is.

### Inference

For the sentiment prediction I've used the [roBERTa-base model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment). This particular implementation was trained on  ~58M tweets and finetuned for tasks such as: sentiment analysis, irony detection, hate detection and others.

explain ...

The inference was performed entirely on Google Colab using the provided GPU, which allowed me to classify 100k tweets in around 5 minutes. Each tweet in the dataset was assigned with the probabilities of belonging to negative, neutral and positive category. To choose the final category, I've simply chosen the class with the highest on. 

### Results

The distribution of the predicted sentiment is as follows:

| Polarity      | Count         |
| ------------- |:-------------:|
| neutral       | 44 719        |
| positive      | 39 632        |
| negative      | 15 519        |

![Sentiment distribution](/img/sentiment.png)

Examples of raw tweets with the highest positive score:

> Excited to share my #career journey @St Albans School and help students think about career choices @Founders4School

> It was wonderful to meet Dame Sian @DameSianP - last night. Thank you Sian for chatting and being so enthusiastic about your work, my work and #bafta #wales

> While I'm a fan of candor, honesty and transparency in business (and life TBH) the 3 people who have suggested the radical candor method to me have always used it as an excuse to act like £$%&holes

While two first examples seems to be assigned correctly, the third one is rather negative. It's good advertiserial example.

Examples of raw tweets with the highest negative score:

> The most important lesson I’ve learned about career advancement is that you need to ask for that promotion

> Your site is appalling and it's impossible to get any information. PLEASE help! This was due weeks ago.

> That is called "science"
> So already done and tested again and again over the last 150 years
> Try to keep up - please

### TO-DO:

- [ ] Some preprocessed examples have lower, correct probability in my pipeline than raw examples in API
- [ ] Improve the text preprocessing - e.g. spelling mistakes, testing more simple pipelines
- [ ] I was unable to fit few tweets to a tokenizer, the prediction is done for 99 870 tweets 
- [ ] Clean and comment the code more 
- [ ] Try other models 

## References:

* http://www.cs.columbia.edu/~julia/papers/Agarwaletal11.pdf
* https://arxiv.org/ftp/arxiv/papers/1601/1601.06971.pdf
* https://arxiv.org/pdf/2010.12421.pdf
* https://github.com/Deffro/text-preprocessing-techniques/blob/master/techniques.py (I copied two re from this repo)
