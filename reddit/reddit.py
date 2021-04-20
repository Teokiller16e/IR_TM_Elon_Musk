import praw
from praw.models import MoreComments

reddit = praw.Reddit(client_id = "retmTurNtUpV4g",
                     client_secret = "ytQWmu0VzUfH4DWicHIALd10EUQHkg",
                     username = "irtmproject2021",
                     password = "irtmproject2021",
                     user_agent="anything")

user = reddit.redditor('ElonMuskOfficial')
submission = reddit.submission(id="590wi9")
submission.comments.replace_more(limit=0)
comment_counter = 0

for comment in submission.comments:

    for reply in comment.replies:
        if reply.author == user:
            print("Comment: ", comment_counter)
            print(comment.body.replace('\n\n', '\n'))
            print("Elon's Reply: ")
            print(reply.body.replace('\n\n', '\n'))
            print()
    comment_counter += 1



