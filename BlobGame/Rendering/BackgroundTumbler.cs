using BlobGame.App;
using BlobGame.ResourceHandling;
using BlobGame.Util;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Graphics.Textures;

namespace BlobGame.Rendering;
/// <summary>
/// Class to handle the logic and drawing of the cute little berries in the background.
/// </summary>
internal sealed class BackgroundTumbler {
    private Texture[] Textures { get; set; }

    private TumblerData[] Tumblers { get; }

    public BackgroundTumbler(int numTumblers) {
        Textures = new Texture[2];

        // Spawn tumblers off screen in a circle around the center of the screen.
        // target a point in a circle around the center of the screen.
        // move in that direction

        Random rng = new Random();
        Tumblers = new TumblerData[numTumblers];
        float cX = GameApplication.PROJECTION_WIDTH / 2f;
        float cY = GameApplication.PROJECTION_HEIGHT / 2f;
        for (int i = 0; i < Tumblers.Length; i++) {
            float angle = rng.NextSingle() * MathF.Tau;
            float r = (0.85f + 1.5f * rng.NextSingle()) * GameApplication.PROJECTION_HEIGHT;
            float pX = MathF.Cos(angle) * r + cX;
            float pY = MathF.Sin(angle) * r + cY;

            float tX = MathF.Cos(rng.NextSingle() * MathF.Tau) * GameApplication.PROJECTION_HEIGHT * 0.4f + cX;
            float tY = MathF.Sin(rng.NextSingle() * MathF.Tau) * GameApplication.PROJECTION_HEIGHT * 0.4f + cY;

            float vX = tX - pX;
            float vY = tY - pY;
            float v = (25 + rng.NextSingle() * 25) / MathF.Sqrt((vX * vX) + (vY * vY));

            Tumblers[i] = new TumblerData(
                rng.Next(2),
                new Vector2(pX, pY),
                new Vector2(vX * v, vY * v),
                rng.NextSingle() * MathF.Tau,
                (0.1f + rng.NextSingle() * 0.4f) * MathF.Tau);
        }
    }

    internal void Load() {
        Textures[0] = ResourceManager.TextureLoader.GetResource($"blueberry_no_face");
        Textures[1] = ResourceManager.TextureLoader.GetResource($"strawberry_no_face");
    }

    internal void Draw(float dT) {
        Vector2 center = new Vector2(GameApplication.PROJECTION_WIDTH / 2f, GameApplication.PROJECTION_HEIGHT / 2f);
        foreach (TumblerData tumbler in Tumblers) {
            float sW = Textures[tumbler.TextureIndex].Width;
            float sH = Textures[tumbler.TextureIndex].Height;

            tumbler.Position += tumbler.Velocity * dT;
            Vector2 dP = tumbler.Position - tumbler.StartPosition;
            Vector2 dCP = tumbler.Position - center;

            if (dP.LengthSquared > GameApplication.PROJECTION_HEIGHT * GameApplication.PROJECTION_HEIGHT &&
                dCP.LengthSquared > GameApplication.PROJECTION_HEIGHT * GameApplication.PROJECTION_HEIGHT)
                tumbler.Position = tumbler.StartPosition;

            Primitives.DrawSprite(
                tumbler.Position,
                new Vector2(sW, sH),
                new Vector2(0.5f, 0.5f),
                (tumbler.Rotation + GameApplication.RenderGameTime * tumbler.AngularVelocity),
                2,
                Textures[tumbler.TextureIndex],
                Color4.White.ChangeAlpha(32));
        }
    }

    private sealed class TumblerData {
        public int TextureIndex { get; }
        public Vector2 StartPosition { get; }
        public Vector2 Position { get; set; }
        public Vector2 Velocity { get; }
        public float Rotation { get; }
        public float AngularVelocity { get; }

        public TumblerData(int textureIndex, Vector2 pos, Vector2 vel, float rot, float angVel) {
            TextureIndex = textureIndex;
            StartPosition = pos;
            Position = pos;
            Velocity = vel;
            Rotation = rot;
            AngularVelocity = angVel;
        }
    }
}
