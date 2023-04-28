import os
import subprocess
import tempfile
import youtube_dl
from google.cloud import storage
from googleapiclient.discovery import build


# Takes search keyword, searches for YouTube videos using keyword and calls scrape_youtube to download each audio
def search_and_download(keyword, api_key):
    # Setting up the YouTube API client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Searches for videos using the provided keyword
    request = youtube.search().list(
        part='id,snippet',
        q=keyword,
        type='video',
        videoDefinition='high',
        maxResults=50
    )
    response = request.execute()

    # Lists video IDs from search results
    video_ids = []
    for item in response['items']:
        video_id = item['id']['videoId']
        video_ids.append(video_id)

    # Downloads audio from the first 2 pages of video search results
    for video_id in video_ids[:20]:
        try:
            scrape_youtube(video_id, keyword)
        except youtube_dl.utils.DownloadError as e:
            print(f"Error downloading video {video_id}: {e}")


# Takes the video ID and search keyword, downloads the audio from the YouTube video and saves to GCP bucket
def scrape_youtube(video_id, keyword):
    youtube_root = 'https://www.youtube.com/watch?v='
    bucket_name = 'music_data_4995'

    # Downloads the audio from the YouTube video
    youtube_url = youtube_root + video_id
    temp_dir = tempfile.mkdtemp()
    file_name = video_id + '.mp3'
    temp_file = os.path.join(temp_dir, file_name)
    subprocess.call(
        ['yt-dlp', '--no-warnings', '--extract-audio', '--audio-format', 'mp3', '--audio-quality', '0', '--verbose',
         '-o', temp_file, youtube_url])

    # Uploads the audio file to GCP storage bucket
    folder_name = keyword.replace(' ', '_')
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder_name}/{file_name}")
    blob.upload_from_filename(temp_file)

    # Cleans up temporary directory
    os.remove(temp_file)
    os.rmdir(temp_dir)


# Runs when youtube_scraper.py is run
if __name__ == "__main__":
    keywords = ['dark music', 'somber music', 'gloomy music', 'sad music',
                'bright music', 'happy music', 'cheerful music',
                'techno music', 'night club music', 'party music',
                'calm music', 'peaceful music', 'relaxing music',
                'classical music', 'classic music']

    # Replace "API_KEY" with actual key
    for keyword in keywords:
        search_and_download(keyword, "API_KEY")
