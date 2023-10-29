using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GUIImage {
    public TextureResource Texture { get; set; }
    public ColorResource Tint { get; set; }

    public float Scale { get; set; }
    private Vector2 Position { get; }
    private Vector2 Pivot { get; }

    public GUIImage(Vector2 pos, float scale, TextureResource texture, Vector2? pivot = null)
        : this(pos.X, pos.Y, scale, texture, pivot) {
    }

    public GUIImage(float x, float y, float scale, TextureResource texture, Vector2? pivot = null) {
        Position = new Vector2(x, y);
        Pivot = pivot ?? new Vector2(0, 0);

        Scale = scale;
        Texture = texture;
        Tint = ColorResource.WHITE;
    }

    internal void Draw() {
        float w = Texture.Resource.width * Scale;
        float h = Texture.Resource.height * Scale;

        float x = Position.X - w * Pivot.X;
        float y = Position.Y - h * Pivot.Y;

        Raylib.DrawTexturePro(
            Texture.Resource,
                new Rectangle(0, 0, Texture.Resource.width, Texture.Resource.height),
                new Rectangle(x, y, w, h),
                new Vector2(0, 0),
                0,
                Tint.Resource);
    }

}
