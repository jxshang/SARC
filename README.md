## <font style="color:rgb(31, 35, 40);">SARC: Sentiment-Augmented Deep Role Clustering for Fake News Detection</font>

<font style="color:rgb(31, 35, 40);">  
</font><font style="color:rgb(31, 35, 40);">This project contains the source code for the work "SARC: Sentiment-Augmented Deep Role Clustering for Fake News Detection", which has been accepted by WSDM 2026.</font>

**<font style="color:rgb(31, 35, 40);">Authors</font>**<font style="color:rgb(31, 35, 40);">：Jingqing Wang, Jiaxing Shang*, Rong Xu, Fei Hao, Tianjin Huang and Geyong Min</font>

**<font style="color:rgb(31, 35, 40);">Reference：</font>**<font style="color:rgb(31, 35, 40);">J. Wang, J. Shang, R. Xu, F. Hao, T. Huang and G. Min, "SARC: Sentiment-Augmented Deep Role Clustering for Fake News Detection," in Proceedings of the 2026 ACM Web Conference (WSDM '26), 2026,（accepted）</font>

### <font style="color:rgb(31, 35, 40);">Raw_Dataset</font>
| **Dataset** | **Resource** |
| --- | --- |
| Weibo-comp | [https://www.datafountain.cn/competitions/422](https://www.datafountain.cn/competitions/422) |
| <font style="color:rgb(31, 35, 40);">RumourEval-19</font> | [https://aclanthology.org/S19-2147/](https://aclanthology.org/S19-2147/) |


<font style="color:rgb(31, 35, 40);">You can customize the dataset to your needs. For this work, we processed the datasets and provided the dataset file, you can download the dataset </font><font style="color:rgba(0, 0, 0, 0.85);"> in the folder "data_with_emotion".</font>

### <font style="color:rgba(0, 0, 0, 0.85);">Resource</font>
you can download the embedding files as the following table:

| **embedding file** | **link** |
| --- | --- |
| glove.42B.300d | [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) |
| sgns.weibo.word | [https://github.com/Embedding/Chinese-Word-Vectors.git](https://nlp.stanford.edu/projects/glove/) |


<font style="color:rgb(31, 35, 40);">After downloading the </font>**embedding file**<font style="color:rgb(31, 35, 40);">, please ensure to place it in the appropriate folder as demonstrated below:</font>

```bash
-resource
   --embedding
     ---glove.42B.300d.txt
     ---sgns.weibo.word
       ----sgns.weibo.word
```

### <font style="color:rgba(0, 0, 0, 0.85);">Code</font>
**<font style="color:rgb(31, 35, 40);">Requirements</font>**<font style="color:rgb(31, 35, 40);">:</font>

```bash
Python=3.8.20
torch=1.5.1
torchvision=0.6.1_cu102
nltk=3.9.1
numpy=1.24.3
```



you can train model by running train.py

