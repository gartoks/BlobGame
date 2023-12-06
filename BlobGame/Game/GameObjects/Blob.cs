using BlobGame.Game.Blobs;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using nkast.Aether.Physics2D.Common;
using nkast.Aether.Physics2D.Dynamics;
using Raylib_CsLo;

namespace BlobGame.Game.GameObjects;

/// <summary>
/// Class for all blobs. 
/// </summary>
internal sealed class Blob : GameObject {
    /// <summary>
    /// The type of the blob.
    /// </summary>
    public int Type { get; }

    /// <summary>
    /// The data of the blob.
    /// </summary>
    public BlobData Data { get; }

    /// <summary>
    /// The physics engine fixture attached to the blob's body.
    /// </summary>
    protected override Fixture Fixture { get; }

    /// <summary>
    /// The texture of the blob.
    /// </summary>
    private TextureResource Texture { get; }
    /// <summary>
    /// The origin of the texuture. Rotation is applied around this point. It is in relative texture coordinates from [0, 1].
    /// </summary>
    private Vector2 TextureOrigin { get; }

    /// <summary>
    /// Create a new blob with the given parameters.
    /// </summary>
    public Blob(BlobData data, World world, Vector2 position, float rotation)
        : base(data.Name, world, position, 0, BodyType.Dynamic) {

        Type = data.Id;
        Data = data;

        Fixture = data.CreateFixture(Body);
        Fixture.Restitution = 0.15f;
        Fixture.Friction = 0.1f;
        Body.Mass = data.Mass;
        Body.AngularDamping = 0.9f;
        Body.Rotation = rotation;

        Texture = ResourceManager.TextureLoader.Get(data.TextureKey);
        TextureOrigin = data.Origin;
    }

    /// <summary>
    /// Custom draw logic for the blob.
    /// </summary>
    protected internal override void DrawInternal() {
        Texture.Draw(
            System.Numerics.Vector2.Zero,
            new System.Numerics.Vector2(TextureOrigin.X, TextureOrigin.Y),
            new System.Numerics.Vector2(Data.TextureScale.X, Data.TextureScale.Y),
            0);

        if (Application.DRAW_DEBUG) {
            if (Data.AsCircle(out float radius))
                Raylib.DrawCircleLines(0, 0, radius * POSITION_MULTIPLIER, Raylib.BLUE);
            else if (Data.AsPolygon(out Vertices vertices)) {
                for (int i = 0; i < vertices.Count; i++) {
                    Vector2 start = vertices[i];
                    Vector2 end = vertices[(i + 1) % vertices.Count];
                    Raylib.DrawLineV(new System.Numerics.Vector2(start.X, start.Y) * POSITION_MULTIPLIER, new System.Numerics.Vector2(end.X, end.Y) * POSITION_MULTIPLIER, Raylib.BLUE);
                }
            }
            Raylib.DrawCircleV(new System.Numerics.Vector2(0, 0), 5f, Raylib.LIME);
        }
    }
}
