"""繪製 online（推薦探索）流程圖，風格對齊 流程圖.png（離線訓練流程）。輸出至腳本所在目錄。"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "online_流程圖.png")

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

FIG_W, FIG_H = 16, 8
XLIM, YLIM = 140, 70


def box(ax, x, y, w, h, text, fontsize=11.5):
    """x,y,w,h：box 左下角座標與寬高。text：box 內文字。回傳 box 座標字典供箭頭定位用。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.6",
        linewidth=1.4, edgecolor="white", facecolor="black", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             color="white", fontsize=fontsize, zorder=4, linespacing=1.4)
    return {"x": x, "y": y, "w": w, "h": h}


def section(ax, x, y, w, h, title):
    """x,y,w,h：外框左下角座標與寬高。title：區塊標題（含編號）。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
        linewidth=1.4, edgecolor="white", facecolor="none",
        linestyle=(0, (5, 3)), zorder=1))
    ax.text(x + 1.2, y + h - 1.6, title, ha="left", va="top",
             color="white", fontsize=14, fontweight="bold", zorder=4)


def pt(b, side, offset=0.0):
    """b：box 座標字典。side：'left'/'right'/'top'/'bottom'。offset：沿該邊方向的位移，避免多條線疊在同一點。"""
    if side == "right":
        return (b["x"] + b["w"], b["y"] + b["h"] / 2 + offset)
    if side == "left":
        return (b["x"], b["y"] + b["h"] / 2 + offset)
    if side == "top":
        return (b["x"] + b["w"] / 2 + offset, b["y"] + b["h"])
    return (b["x"] + b["w"] / 2 + offset, b["y"])


def arrow(ax, p1, p2, rad=0.0, color="white", lw=1.3, ls="-"):
    """p1,p2：箭頭起訖座標。rad：彎曲程度（0 為直線）。"""
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
        color=color, linestyle=ls, shrinkA=2, shrinkB=2, zorder=2,
        connectionstyle=f"arc3,rad={rad}"))


def elbow(ax, p1, p2, via_y, color="white", lw=1.3, ls="-"):
    """p1,p2：箭頭起訖座標。via_y：中間水平轉折的 y 高度，走空白區避免穿過其他方框。"""
    mid1, mid2 = (p1[0], via_y), (p2[0], via_y)
    common = dict(color=color, linewidth=lw, linestyle=ls, zorder=2)
    ax.add_patch(FancyArrowPatch(p1, mid1, arrowstyle="-", shrinkA=2, shrinkB=0, **common))
    ax.add_patch(FancyArrowPatch(mid1, mid2, arrowstyle="-", shrinkA=0, shrinkB=0, **common))
    ax.add_patch(FancyArrowPatch(mid2, p2, arrowstyle="-|>", mutation_scale=14,
                                  shrinkA=0, shrinkB=2, **common))


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.axis("off")

    # 1. Data Selection
    section(ax, 3, 54, 134, 14, "1. Data Selection")
    b1 = box(ax, 7, 57, 32, 6, "Dataset\n(All Check-in Vectors)")
    b2 = box(ax, 48, 57, 32, 6, "User Selects\na Data Point")
    b3 = box(ax, 89, 57, 34, 6, "Category Count Vector\n(Baseline)")
    arrow(ax, pt(b1, "right"), pt(b2, "left"))
    arrow(ax, pt(b2, "right"), pt(b3, "left"))

    # 2. Encode Baseline（narrow, left）
    section(ax, 3, 19, 38, 33, "2. Encode Baseline")
    e1 = box(ax, 8, 40, 28, 5.5, "Trained Encoder\n(from Offline Model)", 10.5)
    e2 = box(ax, 8, 31, 28, 5.5, "Baseline z\n(latent space)", 10.5)
    e3 = box(ax, 8, 22, 28, 5.5, "Baseline Score", 10.5)
    arrow(ax, pt(e1, "bottom"), pt(e2, "top"))
    arrow(ax, pt(e2, "bottom"), pt(e3, "top"))

    # 3. Interactive POI Exploration（wide, right, with loop）
    section(ax, 43, 19, 94, 33, "3. Interactive POI Exploration (User Loop)")
    w = 16.5
    xs = [47, 65.5, 84, 102.5, 121]
    labels = ["User Selects\nPOI(s) to Add", "Updated\nCategory Vector",
              "Trained Encoder\n(reuse)", "New z\n(latent space)", "New Score"]
    boxes3 = [box(ax, x, 39, w, 6, t, 10) for x, t in zip(xs, labels)]
    for i in range(len(boxes3) - 1):
        arrow(ax, pt(boxes3[i], "right"), pt(boxes3[i + 1], "left"))
    arrow(ax, pt(boxes3[-1], "bottom"), pt(boxes3[0], "bottom"), rad=-0.15)
    ax.text((xs[0] + xs[-1]) / 2 + w / 2, 30, "重複探索 Repeat",
             ha="center", va="center", color="white", fontsize=9.5, zorder=4)

    # 1 -> 2 / 3（同一起點分岔，避免互相交叉）
    arrow(ax, pt(b3, "bottom"), pt(e1, "top"), rad=0.2)
    arrow(ax, pt(b3, "bottom"), pt(boxes3[1], "top"), rad=-0.2)

    # 4. Visualization & Decision
    section(ax, 3, 2, 134, 14, "4. Visualization & Decision")
    v1 = box(ax, 7, 5, 27, 6, "Score Trajectory\n(Baseline vs New)", 10.5)
    v2 = box(ax, 38, 5, 27, 6, "Latent Space\nTrajectory", 10.5)
    v3 = box(ax, 69, 5, 27, 6, "Display to User", 10.5)
    v4 = box(ax, 100, 5, 33, 6, "User Decision\n(Adopt / Continue / Stop)", 10.5)
    arrow(ax, pt(v1, "right"), pt(v3, "left"), rad=0.15)
    arrow(ax, pt(v2, "right"), pt(v3, "left"), rad=-0.15)
    arrow(ax, pt(v3, "right"), pt(v4, "left"))

    # 2/3 -> 4：section2 距離近直接拉，section3 距離遠改走底部空白區的直角轉折線
    arrow(ax, pt(e3, "bottom"), pt(v1, "top", offset=-3), rad=0.05)
    arrow(ax, pt(e2, "bottom"), pt(v2, "top", offset=-3), rad=0.05)
    elbow(ax, pt(boxes3[-1], "bottom"), pt(v1, "top", offset=3), via_y=17.6)
    elbow(ax, pt(boxes3[3], "bottom"), pt(v2, "top", offset=3), via_y=17.0)

    fig.savefig(OUT, dpi=150, facecolor="black")
    print(f"已存 {OUT}")


main()
