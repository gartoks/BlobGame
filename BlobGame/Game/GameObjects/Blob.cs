using BlobGame.ResourceHandling;
using nkast.Aether.Physics2D.Common;
using nkast.Aether.Physics2D.Dynamics;
using Raylib_CsLo;

namespace BlobGame.Game.GameObjects;

internal class Blob : GameObject {
    public static Blob Create(World world, Vector2 position, eBlobType blobType) {
        (string name, eBlobType type, int score, float radius, string textureKey) = BlobData.Data.Single(d => d.type == blobType);

        return new Blob(name, type, score, world, position, radius, ResourceHandler.GetTexture(textureKey), new Vector2(0.5f, 0.5f));

        /*return type switch {
            eBlobType.Cherry => new Cherry(world, position),
            eBlobType.Strawberry => new Strawberry(world, position),
            eBlobType.Grape => new Grape(world, position),
            eBlobType.Orange => new Orange(world, position),
            eBlgitobType.Tomato => new Tomato(world, position),
            eBlobType.Apple => new Apple(world, position),
            eBlobType.Yuzu => new Yuzu(world, position),
            eBlobType.Peach => new Peach(world, position),
            eBlobType.Pineapple => new Pineapple(world, position),
            eBlobType.Honeydew => new Honeydew(world, position),
            eBlobType.Watermelon => new Watermelon(world, position),
            _ => throw new NotImplementedException($"Unknown fruit type '{type}'"),
        };*/
    }

    public eBlobType Type { get; }
    public int Score { get; }
    public float Radius { get; }

    protected override Fixture Fixture { get; }

    private TextureResource Texture { get; }
    private Vector2 TextureOrigin { get; }

    protected Blob(string name, eBlobType type, int score, World world, Vector2 position, float radius, TextureResource texture, Vector2 textureOrigin)
        : base(name, world, position, 0, BodyType.Dynamic) {

        Type = type;
        Score = score;
        Radius = radius;

        Fixture = Body.CreateCircle(radius / 10f, 1);
        Fixture.Restitution = 0.15f;
        Fixture.Friction = 0.1f;
        Body.Mass = 1;
        Body.AngularDamping = 0.9f;

        Texture = texture;
        TextureOrigin = textureOrigin;
    }

    protected internal override void DrawInternal() {
        float w = Texture.Resource.width;
        float h = Texture.Resource.height;

        Raylib.DrawTexturePro(
            Texture.Resource,
            new Rectangle(0, 0, w, h),
            new Rectangle(0, 0, w, h),
            new System.Numerics.Vector2(TextureOrigin.X * w, TextureOrigin.Y * h),
            RayMath.RAD2DEG * Rotation,
            Raylib.WHITE);

        if (RaylibApp.DRAW_DEBUG) {
            Raylib.DrawCircleLines(0, 0, Radius, Raylib.BLUE);
            Raylib.DrawCircleV(new System.Numerics.Vector2(0, 0), 5f, Raylib.LIME);
        }
    }
}
