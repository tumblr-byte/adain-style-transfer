
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T, models
from PIL import Image
import cv2
import os


def calc_mean_std(features, eps=1e-6):
    batch_size, c = features.size()[:2]
    features_reshaped = features.reshape(batch_size, c, -1)
    features_mean = features_reshaped.mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features_reshaped.std(dim=2).reshape(batch_size, c, 1, 1) + eps
    return features_mean, features_std

def adain(content_features, style_features):
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features

def denormalize(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).to(device)
    tensor = torch.clamp(tensor * std + mean, 0, 1)
    return tensor

def tensor_to_image(tensor, device):
    image = denormalize(tensor, device).cpu().squeeze(0).permute(1,2,0).numpy()
    image = (image * 255).astype('uint8')
    return image




class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = vgg[:2]
        self.slice2 = vgg[2:7]
        self.slice3 = vgg[7:12]
        self.slice4 = vgg[12:21]
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, x, output_last_feature=False):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h4 if output_last_feature else (h1,h2,h3,h4)

class RC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.activated = activated
    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        return F.relu(h) if self.activated else h

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(512,256)
        self.rc2 = RC(256,256)
        self.rc3 = RC(256,256)
        self.rc4 = RC(256,256)
        self.rc5 = RC(256,128)
        self.rc6 = RC(128,128)
        self.rc7 = RC(128,64)
        self.rc8 = RC(64,64)
        self.rc9 = RC(64,3,activated=False)
    def forward(self, x):
        h = self.rc1(x)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)
        return h

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = Decoder()
    def generate(self, content_images, style_images, alpha=1.0):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)
        return out


# Upload your trained model path
def load_model_weights(model_path='model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Using untrained model.")
    return model.to(device), device

model, device = load_model_weights('model.pth')

# Convert image to tensor with transforms
def load_image_tensor(image_path, resize=512):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0).to(device)


# Apply style transfer on a single image
def stylize_image(content_path, style_path, output_path, alpha=1.0):
    content = load_image_tensor(content_path)
    style = load_image_tensor(style_path)
    with torch.no_grad():
        out = model.generate(content, style, alpha=alpha)
    img_out = tensor_to_image(out, device)
    cv2.imwrite(output_path, cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
    print(f"Stylized image saved at {output_path}")


# Apply style transfer on a video (frame by frame)
def stylize_video(content_video_path, style_path, output_video_path, alpha=1.0):
    style = load_image_tensor(style_path)
    cap = cv2.VideoCapture(content_video_path)
    if not cap.isOpened():
        print("Cannot open video")
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width,height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx +=1
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = T.Compose([
            T.Resize((height,width)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        content_tensor = transform(frame_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out_tensor = model.generate(content_tensor, style, alpha=alpha)
        out_frame = tensor_to_image(out_tensor, device)
        out.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    cap.release()
    out.release()
    print(f"Stylized video saved at {output_video_path}")



# Automatically detect if content is image or video and apply style transfer
def stylize(content_path, style_path, output_path, alpha=1.0):
    ext = os.path.splitext(content_path)[1].lower()
    if ext in ['.jpg','.jpeg','.png','.bmp']:
        stylize_image(content_path, style_path, output_path, alpha)
    elif ext in ['.mp4','.avi','.mov','.mkv']:
        stylize_video(content_path, style_path, output_path, alpha)
    else:
        print("Unsupported file type. Use image or video.")



if __name__=="__main__":
    content_file = "/content/1.mp4" # Path to your content file (image or video)
    style_file = "/content/download (3).jpg"  # Path to your style image
    output_file = "/content/output_result.jpg"  # Path to save output (jpg for image, mp4 for video)
    stylize(content_file, style_file, output_file, alpha=0.8)
