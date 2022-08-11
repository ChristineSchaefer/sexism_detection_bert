import re


def normalize(string):
    reg = r"MENTION[A-Za-z0-9_]*"
    string = remove_emojis(string)
    string = remove_hashtags(string)
    string = remove_mentions(string)
    string = remove_links(string)
    string = remove_word(string, reg)
    return string


def remove_emojis(string):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', string)


def remove_hashtags(string):
    clean_tweet = re.sub(r"#[A-Za-z0-9_]+", "", string)
    clean_tweet = re.sub(r' +', ' ', clean_tweet)
    return clean_tweet.strip()


def remove_mentions(string):
    clean_tweet = re.sub(r"@[A-Za-z0-9_]+", "", string)
    clean_tweet = re.sub(r' +', ' ', clean_tweet)
    return clean_tweet.strip()


def remove_links(string):
    clean_tweet = re.sub(r"https?://\S+", "", string)
    clean_tweet = re.sub(r' +', ' ', clean_tweet)
    return clean_tweet.strip()


def remove_word(string, regex):
    clean_tweet = re.sub(regex, "", string)
    clean_tweet = re.sub(r' +', ' ', clean_tweet)
    return clean_tweet.strip()


def set_to_string(data):
    return str(data)


if __name__ == '__main__':
    reg = r"MENTION[A-Za-z0-9_]*"
    tweet_1 = "MENTION3074 Was it Marylin M who said that a woman who aspires to be as good as man lacks ambition...!? ;)"
    tweet_2 = "MENTION1358 I get where you are coming from, and it's annoying, to be sure. But it's not considered harassment under Twitter's ToS."
    text = u'This is a smiley face \U0001f602'
    link = "This is a text with a URL https://www.java2blog.com/ to remove."
    tweet = "@__therealanna @heyitsrose let's have a zoom meeting tonite! #quarantinelife #girlsnight #onlinehangout"
    no_string = 1234
    print(remove_emojis(text))
    print(remove_hashtags(tweet))
    print(remove_mentions(tweet))
    print(remove_links(link))
    print(remove_word(tweet_1, reg))
    print(remove_word(tweet_2, reg))
    print(type(set_to_string(no_string)))
