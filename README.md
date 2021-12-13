# zisui_maker
書籍を自炊するときに綺麗にするやつ

## Demo

```bash
docker build ./src -t zisuimaker
cat article/pic/syokudou/zisui.png | docker run --rm -i zisuimaker > out.png

```
