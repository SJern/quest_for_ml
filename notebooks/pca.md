# What do Principal Components Actually do Mathematically?

I have recently taken an interest in PCA after watching Professor Gilbert Strang’s SVD lecture. I must have watched at least 15 videos and 7 different blog posts/papers on PCA since. They are all very excellent resources, but I still found myself somewhat unsatisfied. What they do a lot is teaching us what the PCA promise is and why that promise is very useful in Data Science. They teach us how to extract these principal components. Although I don't agree with how some of them do it by applying SVD on the covariance matrix, but that’s not going to be what this post is about. Some of them went the extra mile to prove to us graphically or logically, how the promise has been fulfilled by the principal components. Graphically, a transformed vector can be shown to be clustered with its original group in a plot. Logically, a proof can be expressed in mathematical symbols. My mind was convinced, but my heart was still not. I felt the former approach to be not precise enough. On the other hand, not enough time has been spent to explain what precisely the latter approach is trying to prove. The goal of this post is to 

Graphically vs numerically Vs mathematical proof

What if we rotate the dataset in a way that all its covariance become 0s?

Invent vs discover


```python
def test():
    print('tested.')
```
