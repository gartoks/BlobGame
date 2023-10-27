using BlobGame.ResourceHandling;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Drawing;
/// <summary>
/// Class to handle the logic and drawing of the cute little berries in the background.
/// </summary>
internal sealed class StrawberryBackgroundTumbler {
    private TextureResource[] Textures { get; set; }

    private TumblerData[] Tumblers { get; }

    public StrawberryBackgroundTumbler(int numTumblers) {
        Textures = Enumerable.Range(0, 2).Select(i => ResourceManager.FallbackTexture).ToArray();

        // Spawn tumblers off screen in a circle around the center of the screen.
        // target a point in a circle around the center of the screen.
        // move in that direction

        Random rng = new Random();
        Tumblers = new TumblerData[numTumblers];
        float cX = Application.BASE_WIDTH / 2f;
        float cY = Application.BASE_HEIGHT / 2f;
        for (int i = 0; i < Tumblers.Length; i++) {
            float angle = rng.NextSingle() * MathF.Tau;
            float r = (0.85f + 1.5f * rng.NextSingle()) * Application.BASE_HEIGHT;
            float pX = MathF.Cos(angle) * r + cX;
            float pY = MathF.Sin(angle) * r + cY;

            float tX = MathF.Cos(rng.NextSingle() * MathF.Tau) * Application.BASE_HEIGHT * 0.4f + cX;
            float tY = MathF.Sin(rng.NextSingle() * MathF.Tau) * Application.BASE_HEIGHT * 0.4f + cY;

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
        Textures[0] = ResourceManager.GetTexture($"0");
        Textures[1] = ResourceManager.GetTexture($"1");
    }

    internal void Draw(float dT) {
        foreach (TumblerData tumbler in Tumblers) {
            float sW = Textures[tumbler.TextureIndex].Resource.width;
            float sH = Textures[tumbler.TextureIndex].Resource.height;

            tumbler.Position += tumbler.Velocity * dT;
            Vector2 dP = tumbler.Position - tumbler.StartPosition;

            if (dP.LengthSquared() > Application.BASE_HEIGHT * Application.BASE_HEIGHT)
                tumbler.Position = tumbler.StartPosition;

            Raylib.DrawTexturePro(
                Textures[tumbler.TextureIndex].Resource,
                new Rectangle(0, 0, sW, sH),
                new Rectangle(tumbler.Position.X, tumbler.Position.Y, sW, sH),
                new Vector2(0.5f * sW, 0.5f * sH),
                (tumbler.Rotation + Renderer.Time * tumbler.AngularVelocity) * RayMath.RAD2DEG, Raylib.WHITE.ChangeAlpha(32));
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
