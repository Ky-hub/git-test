from smplx import FLAME
import torch, trimesh

# 1. 加载 FLAME 模型
flame_model = FLAME(model_path="path/to/flame_model", use_face_contour=True)

# 2. 加载 .ply mesh
mesh = trimesh.load('sentence01.000001.ply', process=False)
target_vertices = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0)

# 3. 优化表情参数
expression = torch.zeros((1, 100), requires_grad=True)
optimizer = torch.optim.Adam([expression], lr=0.01)

for i in range(200):
    optimizer.zero_grad()
    pred_verts = flame_model(expression=expression, return_verts=True).vertices
    loss = ((pred_verts - target_vertices)**2).mean()
    loss.backward()
    optimizer.step()

# 得到的 expression 即该帧的表情参数
expr_params = expression.detach().cpu().numpy()



