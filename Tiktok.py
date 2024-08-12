#!/usr/bin/env python
# coding: utf-8

# # TikTok Project
The goal is to construct a dataframe in Python, perform a cursory inspection of the provided dataset, and inform TikTok data team members about the findings
# # Task 1 . Understand the data - Inspect the data
# 
# Consider the following questions:
# 
# Question 1: When reviewing the first few rows of the dataframe, what do you observe about the data? What does each row represent?
# 
# Question 2: When reviewing the data.info() output, what do you notice about the different variables? Are there any null values? Are all of the variables numeric? Does anything else stand out?
# 
# Question 3: When reviewing the data.describe() output, what do you notice about the distributions of each variable? Are there any questionable values? Does it seem that there are outlier values?

# In[1]:


# Import packages
import pandas as pd
import numpy as np


# In[3]:


data = pd.read_csv("tiktok_dataset.csv")


# In[4]:


# Display and examine the first 10 rows of the dataframe

data.head(10)


# In[5]:


# Get summary info

data.info()


# In[6]:


# Get summary statistics

data.describe()

Response:

Question 1: The dataframe contains a collection of categorical, text, and numerical data. Each row represents a distinct TikTok video that presents either a claim or an opinion and the accompanying metadata about that video.

Question 2: The dataframe contains five float64s, three int64s, and four objects. There are 19,382 observations, but some of the variables are missing values, including claim status, the video transcripton, and all of the count variables.

Question 3: Many of the count variables seem to have outliers at the high end of the distribution. They have very large standard deviations and maximum values that are very high compared to their quartile values.
# # Task 2. Understand the data - Investigate the variables
# In this phase, you will begin to investigate the variables more closely to better understand them.
# 
# You know from the project proposal that the ultimate objective is to use machine learning to classify videos as either claims or opinions. A good first step towards understanding the data might therefore be examining the claim_status variable. Begin by determining how many videos there are for each different claim status.

# In[7]:


# What are the different values for claim status and how many of each are in the data?

data['claim_status'].value_counts()

Response: The counts of each claim status are quite balanced.Next, examine the engagement trends associated with each different claim status.

Start by using Boolean masking to filter the data according to claim status, then calculate the mean and median view counts for each claim status.
# In[8]:


# What is the average view count of videos with "claim" status?

claims = data[data['claim_status'] == 'claim']
print('Mean view count claims:', claims['video_view_count'].mean())
print('Median view count claims:', claims['video_view_count'].median())


# In[9]:


# What is the average view count of videos with "opinion" status?

opinions = data[data['claim_status'] == 'opinion']
print('Mean view count opinions:', opinions['video_view_count'].mean())
print('Median view count opinions:', opinions['video_view_count'].median())

Response: The mean and the median within each claim category are close to one another, but there is a vast discrepancy between view counts for videos labeled as claims and videos labeled as opinions.Now, examine trends associated with the ban status of the author.

Use groupby() to calculate how many videos there are for each combination of categories of claim status and author ban status.
# In[10]:


# Get counts for each group combination of claim status and author ban status

data.groupby(['claim_status', 'author_ban_status']).count()[['#']]

Response: There are many more claim videos with banned authors than there are opinion videos with banned authors. This could mean a number of things, including the possibilities that:

Claim videos are more strictly policed than opinion videos
Authors must comply with a stricter set of rules if they post a claim than if they post an opinion
Also, it should be noted that there's no way of knowing if claim videos are inherently more likely than opinion videos to result in author bans, or if authors who post claim videos are more likely to post videos that violate terms of service.Finally, while you can use this data to draw conclusions about banned/active authors, you cannot draw conclusions about banned videos. There's no way of determining whether a particular video caused the ban, and banned authors could have posted videos that complied with the terms of service.

Continue investigating engagement levels, now focusing on author_ban_status.

Calculate the median video share count of each author ban status.
# In[11]:



data.groupby(['author_ban_status']).agg(
    {'video_view_count': ['mean', 'median'],
     'video_like_count': ['mean', 'median'],
     'video_share_count': ['mean', 'median']})


# In[12]:


# What's the median video share count of each author ban status?

data.groupby(['author_ban_status']).median(numeric_only=True)[
    ['video_share_count']]

response: Banned authors have a median share count that's 33 times the median share count of active authors! Explore this in more depth.
# Use groupby() to group the data by author_ban_status, then use agg() to get the count, mean, and median of each of the following columns:
# 
# video_view_count
# video_like_count
# video_share_count
# Remember, the argument for the agg() function is a dictionary whose keys are columns. The values for each column are a list of the calculations you want to perform.

# In[13]:



data.groupby(['author_ban_status']).agg(
    {'video_view_count': ['count', 'mean', 'median'],
     'video_like_count': ['count', 'mean', 'median'],
     'video_share_count': ['count', 'mean', 'median']
     })

response: A few observations stand out:

Banned authors and those under review get far more views, likes, and shares than active authors.
In most groups, the mean is much greater than the median, which indicates that there are some videos with very high engagement counts.Now, create three new columns to help better understand engagement rates:

likes_per_view: represents the number of likes divided by the number of views for each video
comments_per_view: represents the number of comments divided by the number of views for each video
shares_per_view: represents the number of shares divided by the number of views for each video
# In[15]:


# Create a likes_per_view column
data['likes_per_view'] = data['video_like_count'] / data['video_view_count']

# Create a comments_per_view column
data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']

# Create a shares_per_view column
data['shares_per_view'] = data['video_share_count'] / data['video_view_count']


# Use groupby() to compile the information in each of the three newly created columns for each combination of categories of claim status and author ban status, then use agg() to calculate the count, the mean, and the median of each group.

# In[16]:


data.groupby(['claim_status', 'author_ban_status']).agg(
    {'likes_per_view': ['count', 'mean', 'median'],
     'comments_per_view': ['count', 'mean', 'median'],
     'shares_per_view': ['count', 'mean', 'median']})

 response: We know that videos by banned authors and those under review tend to get far more views, likes, and shares than videos by non-banned authors. However, when a video does get viewed, its engagement rate is less related to author ban status and more related to its claim status.

Also, we know that claim videos have a higher view rate than opinion videos, but this tells us that claim videos also have a higher rate of likes on average, so they are more favorably received as well. Furthermore, they receive more engagement via comments and shares than opinion videos.

Note that for claim videos, banned authors have slightly higher likes/view and shares/view rates than active authors or those under review. However, for opinion videos, active authors and those under review both get higher engagement rates than banned authors in all categories.
# # Given your efforts, what can you summarize for Rosie Mae Bradshaw and the TikTok data team?
# Note for Learners: Your answer should address TikTok's request for a summary that covers the following points:
# 
# What percentage of the data is comprised of claims and what percentage is comprised of opinions?
# What factors correlate with a video's claim status?
# What factors correlate with a video's engagement level?

# # response:
# 
# Of the 19,382 samples in this dataset, just under 50% are claims—9,608 of them.
# 
# Engagement level is strongly correlated with claim status. This should be a focus of further inquiry.
# 
# Videos with banned authors have significantly higher engagement than videos with active authors. 
# 
# Videos with authors under review fall between these two categories in terms of engagement levels.
