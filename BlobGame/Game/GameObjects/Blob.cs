using BlobGame.App;
using BlobGame.Game.Blobs;
using BlobGame.ResourceHandling;
using nkast.Aether.Physics2D.Common;
using nkast.Aether.Physics2D.Dynamics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Graphics.Textures;

namespace BlobGame.Game.GameObjects;

/// <summary>
/// Class for all blobs. 
/// </summary>
internal sealed class Blob : GameObject {
    /// <summary>
    /// Creates a new blob with the given parameters and resolves resources.
    /// </summary>
    /// <param name="world"></param>
    /// <param name="position"></param>
    /// <param name="blobType"></param>
    /// <returns></returns>
    public static Blob Create(World world, Vector2 position, eBlobType blobType) {
        (string name, eBlobType type, int score, float radius, float mass, string textureKey) = BlobData.Data.Single(d => d.type == blobType);
        return new Blob(name, type, score, world, position, radius, mass, ResourceManager.TextureLoader.GetResource(textureKey), new Vector2(0.5f, 0.5f));
    }

    /// <summary>
    /// The type of the blob.
    /// </summary>
    public eBlobType Type { get; }
    /// <summary>
    /// The score gained when two blobs of this type collide and combine.
    /// </summary>
    public int Score { get; }
    /// <summary>
    /// The radius of the blob's circle collider.
    /// </summary>
    public float Radius { get; }

    /// <summary>
    /// The physics engine fixture attached to the blob's body.
    /// </summary>
    protected override Fixture Fixture { get; }

    /// <summary>
    /// The texture of the blob.
    /// </summary>
    private Texture Texture { get; }
    /// <summary>
    /// The origin of the texuture. Rotation is applied around this point. It is in relative texture coordinates from [0, 1].
    /// </summary>
    private Vector2 TextureOrigin { get; }

    /// <summary>
    /// Create a new blob with the given parameters.
    /// </summary>
    private Blob(string name, eBlobType type, int score, World world, Vector2 position, float radius, float mass, Texture texture, Vector2 textureOrigin)
        : base(name, world, position, 0, BodyType.Dynamic) {

        Type = type;
        Score = score;
        Radius = radius;

        Fixture = Body.CreateCircle(radius / 10f, 1);
        Fixture.Restitution = 0.15f;
        Fixture.Friction = 0.1f;
        Body.Mass = mass;
        Body.AngularDamping = 0.9f;

        Texture = texture;
        TextureOrigin = textureOrigin;
    }

    /// <summary>
    /// Custom draw logic for the blob.
    /// </summary>
    public override void Render(OpenTK.Mathematics.Vector2 offset) {
        OpenTK.Mathematics.Vector2 pos = new OpenTK.Mathematics.Vector2(Position.X, Position.Y) + offset;
        Primitives.DrawSprite(pos, new OpenTK.Mathematics.Vector2(Radius, Radius), new OpenTK.Mathematics.Vector2(0.5f, 0.5f), Rotation, 4, Texture, OpenTK.Mathematics.Color4.White);

        if (GameApplication.DRAW_DEBUG) {
            Primitives.DrawCircleLines(pos, Radius, new OpenTK.Mathematics.Vector2(0.5f, 0.5f), int.MaxValue, OpenTK.Mathematics.Color4.Blue);
            Primitives.DrawCircle(pos, 5, new OpenTK.Mathematics.Vector2(0.5f, 0.5f), int.MaxValue, OpenTK.Mathematics.Color4.Lime);
        }
    }
}
