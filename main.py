from telebot import TeleBot
from model import ColorizationModel
from torchvision.transforms import ToTensor
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torch
import numpy as np
import generator_meme

token = "5016618436:AAHLdkuP94x6p1BgcibbtDWRRt_irqnbULc"
bot = TeleBot(token, parse_mode=None)


@bot.message_handler(commands=["start", "help"])
def start_handler(message):
    message_text = "Привет. Я бот для раскрашивания картинок. Пришлите мне черно-белое изображение, и я конвертирую его в цветное."
    chat_id = message.chat.id
    bot.send_message(chat_id, message_text)


@bot.message_handler(commands=["meme"])
def meme_handler(message):
    path = "Memes/"+ generator_meme.random_meme()
    out_img = open(path, "rb")
    bot.send_photo(message.chat.id, out_img)


@bot.message_handler(content_types=["text", "sticker", "pinned_message", "audio"])
def text_handler(message):
    message_text = "Пожалуйста, пришлите мне фото"
    bot.reply_to(message, message_text)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ColorizationModel()
model.load_state_dict(torch.load("generator80.pt"))
model = model.to(device)
model.eval()


def colorization_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1.
    L = L.unsqueeze(0).to(device)
    ab = model(L)
    L = (L + 1.) * 50.
    ab *= 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().detach().numpy()
    Lab = np.squeeze(Lab)
    img_rgb = lab2rgb(Lab)
    del img
    del img_lab
    del L
    del ab
    del Lab
    torch.cuda.empty_cache()
    return img_rgb


@bot.message_handler(content_types=["photo"])
def photo_handler(message):
    raw = message.photo[-1].file_id
    name = "images/" + raw + ".jpg"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(name, "wb") as f:
        f.write(downloaded_file)
    chat_id = message.chat.id
    bot.send_message(chat_id, "Подождите...")

    try:
        c_img = colorization_img(name)
        c_img = c_img * 255
        c_img = c_img.astype(np.uint8)
        c_img = Image.fromarray(c_img).convert("RGB")
        c_name = "colorization/" + raw + ".jpg"
        c_img.save(c_name)

        out_img = open(c_name, "rb")
        bot.send_message(chat_id, "RGB image")
        bot.send_photo(chat_id, out_img)
    except Exception as e:
        print(e)
        bot.send_message(chat_id, "Что-то пошло не так")


if __name__ == "__main__":
    bot.infinity_polling()
