import imageio.v3 as iio
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, video_paths: list[str]):
        self.video_paths = video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        video_path = self.video_paths[idx]

        frames = []
        for frame in iio.imiter(video_path, plugin="pyav"):
            frame = torch.from_numpy(frame).permute(2, 0, 1).float().div(255)
            frames.append(frame)

            if len(frames) == 17:
                break

        frames = torch.stack(frames, dim=1)

        return {"pixel_values": frames}
