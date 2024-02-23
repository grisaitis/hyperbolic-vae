import io

import imageio.v3 as iio
import plotly.express as px

fig = px.scatter(
    px.data.iris(),
    x="sepal_length",
    y="sepal_width",
    color="species",
)
buf = io.BytesIO()

fig.write_image(
    buf,
    engine="kaleido",
    format="png",
    scale=2,
    width=800,
    height=600,
)

buf.seek(0)

im = iio.imread(buf)
im = im.transpose(2, 0, 1)
print("image shape:", im.shape)
print("image type:", type(im))
