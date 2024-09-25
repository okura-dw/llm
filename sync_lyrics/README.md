# sync_lyrics

~~動画IDと歌詞を入力~~音声ファイルと歌詞を入力したら[SubRipフォーマット](https://ja.wikipedia.org/wiki/SubRip#SubRip%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%83%95%E3%82%A9%E3%83%BC%E3%83%9E%E3%83%83%E3%83%88)で歌詞が表示されるシステム

> [!WARNING]
> 09/25現在、動画IDから楽曲の音声ファイルを取得する機構がないため、__動画IDと歌詞を入力__ではなく__音声ファイルと歌詞を入力__になっています。

## 目的

音楽に同期して歌詞を自動で流すために、歌詞にタイムスタンプをつけたい

## 使い方

```
$ mkdir -p downloads/{music,lyrics}
downloads/music 以下に曲の音声ファイル（曲名.wav等）を、downloads/lyrics 以下に歌詞のテキストファイル（曲名.txt）を配置する
$ pip install -r requirements.txt
$ streamlit run main.py
```