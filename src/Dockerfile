FROM python:3.6

WORKDIR /work
RUN mkdir /work/sample /work/pic

RUN apt-get update && apt-get install -y libgl1-mesa-dev

COPY ./ /work
RUN pip install -r requirements.txt

# ホストから標準入力で画像をもらい標準出力で画像を出力する
# ログに使用していた標準出力はエラー退避
CMD sh -c "cat > ./sample/zisui.png && python main.py 1>2 && cat ./pic/output.png"