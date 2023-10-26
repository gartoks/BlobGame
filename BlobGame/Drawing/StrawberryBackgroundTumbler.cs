using BlobGame.ResourceHandling;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Drawing;
/// <summary>
/// Class to handle the logic and drawing of the cute little strawberries in the background.
/// </summary>
internal sealed class StrawberryBackgroundTumbler {
    private TextureResource StrawberryTexture { get; set; }

    private TumblerData[] Tumblers { get; }

    public StrawberryBackgroundTumbler(int numTumblers) {
        StrawberryTexture = ResourceManager.DefaultTexture;

        // Spawn tumblers off screen in a circle around the center of the screen.
        // target a point in a circle around the center of the screen.
        // move in that direction

        Random rng = new Random();
        Tumblers = new TumblerData[numTumblers];
        float cX = Application.BASE_WIDTH / 2f;
        float cY = Application.BASE_HEIGHT / 2f;
        for (int i = 0; i < Tumblers.Length; i++) {
            float angle = rng.NextSingle() * MathF.Tau;
            float r = (1f + 1.5f * rng.NextSingle()) * Application.BASE_HEIGHT;
            float pX = MathF.Cos(angle) * r + cX;
            float pY = MathF.Sin(angle) * r + cY;

            float tX = MathF.Cos(rng.NextSingle() * MathF.Tau) * Application.BASE_HEIGHT * 0.4f + cX;
            float tY = MathF.Sin(rng.NextSingle() * MathF.Tau) * Application.BASE_HEIGHT * 0.4f + cY;

            float vX = tX - pX;
            float vY = tY - pY;
            float v = (25 + rng.NextSingle() * 25) / MathF.Sqrt((vX * vX) + (vY * vY));

            Tumblers[i] = new TumblerData(
                new Vector2(pX, pY),
                new Vector2(vX * v, vY * v),
                rng.NextSingle() * MathF.Tau,
                (0.1f + rng.NextSingle() * 0.4f) * MathF.Tau);
        }
    }

    internal void Load() {
        StrawberryTexture = ResourceManager.GetTexture($"1");
    }

    internal void Draw() {
        float sW = StrawberryTexture.Resource.width;
        float sH = StrawberryTexture.Resource.height;

        foreach (TumblerData tumbler in Tumblers) {
            float x = tumbler.Position.X + tumbler.Velocity.X * Renderer.Time;
            float y = tumbler.Position.Y + tumbler.Velocity.Y * Renderer.Time;

            float dx = x - tumbler.StartPosition.X;
            float dy = y - tumbler.StartPosition.Y;

            if (dx * dx + dy * dy > Application.BASE_HEIGHT * Application.BASE_HEIGHT)
                tumbler.Position = tumbler.StartPosition;

            Raylib.DrawTexturePro(
                StrawberryTexture.Resource,
                new Rectangle(0, 0, sW, sH),
                new Rectangle(x, y, sW, sH),
                new Vector2(0.5f * sW, 0.5f * sH),
                (tumbler.Rotation + Renderer.Time * tumbler.AngularVelocity) * RayMath.RAD2DEG, Raylib.WHITE.ChangeAlpha(32));
        }
    }

    private sealed class TumblerData {
        public Vector2 StartPosition { get; set; }
        public Vector2 Position { get; set; }
        public Vector2 Velocity { get; set; }
        public float Rotation { get; set; }
        public float AngularVelocity { get; set; }

        public TumblerData(Vector2 pos, Vector2 vel, float rot, float angVel) {
            StartPosition = pos;
            Position = pos;
            Velocity = vel;
            Rotation = rot;
            AngularVelocity = angVel;
        }
    }
}
