using BlobGame.ResourceHandling.Resources;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GUIImage : GuiElement {
    public TextureResource Texture { get; set; }
    public ColorResource Tint { get; set; }

    public float Scale { get; set; }

    public GUIImage(float x, float y, float scale, TextureResource texture, Vector2? pivot = null)
        : base(x, y, scale * texture.Resource.width, scale * texture.Resource.height, pivot) {

        Scale = scale;
        Texture = texture;
        Tint = ColorResource.WHITE;
    }

    protected override void DrawInternal() {
        Texture.Draw(new Vector2(Bounds.x, Bounds.y), Pivot, new Vector2(Scale, Scale), 0, Tint.Resource);

        //Raylib.DrawTexturePro(
        //    Texture.Resource,
        //        new Rectangle(0, 0, Texture.Resource.width, Texture.Resource.height),
        //        Bounds,
        //        Pivot,
        //        0,
        //        Tint.Resource);
    }

}
