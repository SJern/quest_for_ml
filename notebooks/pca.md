# What do Principal Components Actually do Mathematically?
I have recently taken an interest in PCA after watching Professor Gilbert Strang’s [SVD lecture](https://www.youtube.com/watch?v=rYz83XPxiZo). I must have watched at least 15 other videos and read 7 different blog posts on PCA since. They are all very excellent resources, but I found myself somewhat unsatisfied. What they do a lot is teaching us the following:
- What the PCA promise is;
- Why that promise is very useful in Data Science; and
- How to extract these principal components. (Although I don't agree with how some of them do it by applying SVD on the covariance matrix, that can be saved for another post.)

Some of them go the extra mile to show us graphically or logically, how the promise is being fulfilled by the principal components:
- Graphically, a transformed vector can be shown to be still clustered with its original group in a plot; and
- Logically, a proof can be expressed in mathematical symbols.

## Objective
My mind was convinced, but my heart was still not. To me, the graphic approach does not provide a visual effect that is striking enough. The logic approach, on the other hand, does not even spend enough effort to explain what precisely it's trying to prove. Therefore, the objective of this post is to improve both of these 2 areas - to provide a more striking visual and to establish a more precise goal for our mathematical proof.

## Prerequisites
This post is for you if:
- You have at least a slight idea of what the PCA promise is;
- You have a decent understanding of what the covariance matrix is about;
- You have a solid foundation in linear algebra, e.g., comfortable with the concept of matrices being transformations that rotate and/or stretch spaces; and
- Your heart is longing to discover the principal components, instead of being told what they are!

## How to Choose P?
After hearing my dissatisfaction, my friend [Calvin](https://calvinfeng.github.io/) recommended this paper by Jonathon Shlens - [A Tutorial on Principal Component Analysis](https://arxiv.org/pdf/1404.1100.pdf) to me. It is by far the best resource I have come across on PCA. However, it's also a bit lengthier than your typical blog post, so the remainder of this post will focus on section 5 of the paper. In there, Jonathon immediately establishes the following goal:
> The [original] dataset is $$X$$, an $$m × n$$ matrix.<br>
> Find some orthonormal matrix $$P$$ in $$Y = PX$$ such that $$C_Y \equiv \frac{1}{n}YY^T$$ is a diagonal matrix.[1]<br>
> The rows of $$P$$ shall be principal components of $$X$$.

As you might have noticed, $$C_Y$$ here is the covariance matrix of our rotated dataset $$Y$$. Why do we want $$C_Y$$ to be diagonal? Before we answer this question, let’s generate a dataset $$X$$ consisting of 4 features with some random values.


```python
import numpy as np
feature_num, sample_num = 4, 6
x = np.random.rand(feature_num, sample_num)

def covariance_matrix(dataset):
    sample_count = dataset.shape[1]
    return (dataset @ dataset.transpose() / sample_count).round(15)
```


```python
c_x = covariance_matrix(x)
_, p = np.linalg.eig(c_x)
y = p.transpose() @ x
c_y = covariance_matrix(y)
```


```python
from IPython.display import display
import pandas as pd

display(pd.DataFrame.from_records(x, columns=['sample{}'.format(num) for num in range(sample_num)], index=['feat_{}'.format(num) for num in 'abcd']))
display(pd.DataFrame.from_records(y, columns=['sample{}'.format(num) for num in range(sample_num)], index=['feat_{}'.format(num) for num in 'efgh']))
display(pd.DataFrame.from_records(c_x, columns=['feat_{}'.format(num) for num in 'abcd'], index=['feat__{}'.format(num) for num in 'abcd']))
display(pd.DataFrame.from_records(c_y, columns=['feat_{}'.format(num) for num in 'efgh'], index=['feat__{}'.format(num) for num in 'efgh']))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample0</th>
      <th>sample1</th>
      <th>sample2</th>
      <th>sample3</th>
      <th>sample4</th>
      <th>sample5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feat_a</th>
      <td>0.364316</td>
      <td>0.369955</td>
      <td>0.414107</td>
      <td>0.398438</td>
      <td>0.273329</td>
      <td>0.974257</td>
    </tr>
    <tr>
      <th>feat_b</th>
      <td>0.233349</td>
      <td>0.781982</td>
      <td>0.420029</td>
      <td>0.308131</td>
      <td>0.333717</td>
      <td>0.362706</td>
    </tr>
    <tr>
      <th>feat_c</th>
      <td>0.089461</td>
      <td>0.010197</td>
      <td>0.128515</td>
      <td>0.376316</td>
      <td>0.229315</td>
      <td>0.730119</td>
    </tr>
    <tr>
      <th>feat_d</th>
      <td>0.381779</td>
      <td>0.965491</td>
      <td>0.635578</td>
      <td>0.879495</td>
      <td>0.542847</td>
      <td>0.051483</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample0</th>
      <th>sample1</th>
      <th>sample2</th>
      <th>sample3</th>
      <th>sample4</th>
      <th>sample5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feat_e</th>
      <td>0.573571</td>
      <td>1.200266</td>
      <td>0.866370</td>
      <td>1.037611</td>
      <td>0.721114</td>
      <td>0.905122</td>
    </tr>
    <tr>
      <th>feat_f</th>
      <td>0.006382</td>
      <td>-0.446941</td>
      <td>-0.115716</td>
      <td>-0.114439</td>
      <td>-0.075257</td>
      <td>0.890546</td>
    </tr>
    <tr>
      <th>feat_g</th>
      <td>-0.105567</td>
      <td>0.040343</td>
      <td>-0.055121</td>
      <td>0.001167</td>
      <td>0.057684</td>
      <td>0.018866</td>
    </tr>
    <tr>
      <th>feat_h</th>
      <td>-0.027708</td>
      <td>-0.196512</td>
      <td>-0.036851</td>
      <td>0.281227</td>
      <td>0.065940</td>
      <td>-0.061503</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feat_a</th>
      <th>feat_b</th>
      <th>feat_c</th>
      <th>feat_d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feat__a</th>
      <td>0.270619</td>
      <td>0.185934</td>
      <td>0.168921</td>
      <td>0.218072</td>
    </tr>
    <tr>
      <th>feat__b</th>
      <td>0.185934</td>
      <td>0.196706</td>
      <td>0.090022</td>
      <td>0.263646</td>
    </tr>
    <tr>
      <th>feat__c</th>
      <td>0.168921</td>
      <td>0.090022</td>
      <td>0.125316</td>
      <td>0.103120</td>
    </tr>
    <tr>
      <th>feat__d</th>
      <td>0.218072</td>
      <td>0.263646</td>
      <td>0.103120</td>
      <td>0.425455</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feat_e</th>
      <th>feat_f</th>
      <th>feat_g</th>
      <th>feat_h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feat__e</th>
      <td>0.822684</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>feat__f</th>
      <td>0.000000</td>
      <td>0.170837</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>feat__g</th>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.003249</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>feat__h</th>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.021327</td>
    </tr>
  </tbody>
</table>
</div>


[1]: The reason orthonormality is part of the goal is that we do not want to do anything more than rotating $$X$$. We do not want to modify $$X$$. We only want to re-express $$X$$ by carefully choosing a change of basis.
