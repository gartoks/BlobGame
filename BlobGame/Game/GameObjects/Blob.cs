﻿using BlobGame.Game.Blobs;
using BlobGame.ResourceHandling;
using nkast.Aether.Physics2D.Common;
using nkast.Aether.Physics2D.Dynamics;
using Raylib_CsLo;

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
        (string name, eBlobType type, int score, float radius, string textureKey) = BlobData.Data.Single(d => d.type == blobType);
        return new Blob(name, type, score, world, position, radius, ResourceManager.GetTexture(textureKey), new Vector2(0.5f, 0.5f));
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
    private TextureResource Texture { get; }
    /// <summary>
    /// The origin of the texuture. Rotation is applied around this point. It is in relative texture coordinates from [0, 1].
    /// </summary>
    private Vector2 TextureOrigin { get; }

    /// <summary>
    /// Create a new blob with the given parameters.
    /// </summary>
    private Blob(string name, eBlobType type, int score, World world, Vector2 position, float radius, TextureResource texture, Vector2 textureOrigin)
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

    /// <summary>
    /// Custom draw logic for the blob.
    /// </summary>
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

        if (Application.DRAW_DEBUG) {
            Raylib.DrawCircleLines(0, 0, Radius, Raylib.BLUE);
            Raylib.DrawCircleV(new System.Numerics.Vector2(0, 0), 5f, Raylib.LIME);
        }
    }
}
