from models.abstract_model import *
import setup_feed

def custom_summary(feed_url, count, progress_callback, model, prompt,
                   chunk_size, chunk_overlap):    
    feeds = setup_feed.setup_feed(feed_url, count=count)
    summaries = []
    for i, feed in enumerate(feeds):
        content = setup_feed.Content(feed['link'], chunk_size, chunk_overlap)
        summary = model.predict(content, prompt)
        summaries.append({"title": feed["title"], "summary": summary})

        if not progress_callback is None:
            percent = int((i * 100)//count)
            progress_callback(percent)
    
    return summaries