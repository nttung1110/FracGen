# Write a script which receive an input image, a specified pixel location. The output is two videos, one meta data file:
# + First video is called force condition video where the it follows the following 
#
'''
    Write a script which receive an input image, two specified pixel location. The output is two videos, one meta data file:
    + First video is called force condition video, which shows 2 gaussian blobs moving in opposite direction with a fixed velocity
        . Here is a few relevant utils function that is used to save this video:
            def get_gaussian_blob(point, radius=10, amplitude=1.0, shape=(3, 480, 720), device=None):
                x, y = point
                C, H, W = shape

                yy, xx = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij"
                )

                dist2 = (xx - x) ** 2 + (yy - y) ** 2
                gaussian = amplitude * torch.exp(-dist2 / (2.0 * radius ** 2))

                # expand to all channels without loop
                return gaussian.expand(C, H, W)

            u1, v1 = int(u1.detach()), int(v1.detach())

            # blob built at NATIVE rasterizer resolution, resize after accumulation
            blob = get_gaussian_blob(
                [u1, v1],
                shape=(1, rendering.shape[1], rendering.shape[2])  # native res
            ).numpy()
    + Second video is called first-frame video, which is just a single-frame video from the input image.
    + Meta data file is a csv file and is in the following format. Here is an example
        video,prompt,force_video,stress_video,flow_video,damage_video,strain_video
        videos/images.mp4,physical simulation deformation (multimodal-conditioned),force_videos/force_condition.mp4,None,None,None,None

    First video is stored in "{output_path}/force_videos/force_condition.mp4", Second video is in "{output_path}/videos/images.mp4"
    Metadata file is stored in "{output_path}/metadata.csv"
'''