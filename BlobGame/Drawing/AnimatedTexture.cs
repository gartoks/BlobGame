using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Drawing;
/// <summary>
/// Class to handle animated textures.
/// </summary>
internal sealed class AnimatedTexture {
    private TextureResource Texture { get; }

    private Vector2 Pivot { get; }
    private Vector2 Position { get; }
    private Vector2 Scale { get; }
    private float Rotation { get; }
    private Color Color { get; }

    private float Duration { get; }

    internal Func<float, Vector2>? PositionAnimator { get; init; }
    internal Func<float, Vector2>? ScaleAnimator { get; init; }
    internal Func<float, float>? RotationAnimator { get; init; }
    internal Func<float, Color>? ColorAnimator { get; init; }

    /// <summary>
    /// Time when the animation started.
    /// </summary>
    private float StartTime { get; set; }

    /// <summary>
    /// Indicator if the animation has started and finished.
    /// </summary>
    public bool IsFinished => Renderer.Time - StartTime >= Duration;

    /// <summary>
    /// Indicator if the animation has not yet started.
    /// </summary>
    public bool IsReady => Renderer.Time - StartTime < 0;

    public AnimatedTexture(string textureKey, float duration, Vector2 position, Vector2 pivot, Vector2? scale = null, float rotation = 0, Color? color = null)
        : this(ResourceManager.GetTexture(textureKey), duration, position, pivot, scale, rotation, color) {
    }

    public AnimatedTexture(TextureResource texture, float duration, Vector2 position, Vector2? pivot = null, Vector2? scale = null, float rotation = 0, Color? color = null) {
        if (pivot == null)
            pivot = Vector2.Zero;

        if (scale == null)
            scale = Vector2.One;

        if (color == null)
            color = Raylib.WHITE;

        Texture = texture;
        Position = position;
        Pivot = pivot.Value;
        Scale = scale.Value;
        Rotation = rotation;
        Duration = duration;
        Color = color.Value;

        StartTime = -float.MinValue;
    }

    /// <summary>
    /// Starts the animation.
    /// </summary>
    public void Start() {
        StartTime = Renderer.Time;
    }

    /// <summary>
    /// Resets the animation to the "not yet started" state.
    /// </summary>
    public void Reset() {
        StartTime = -float.MinValue;
    }

    public void Draw() {
        float t;
        if (IsReady)
            t = 0;
        else if (IsFinished)
            t = 1;
        else
            t = (Renderer.Time - StartTime) / Duration;

        Vector2 pos = Position + (PositionAnimator?.Invoke(t) ?? Vector2.Zero);
        Vector2 scale = Scale * (ScaleAnimator?.Invoke(t) ?? Vector2.One);
        float rot = (/*Rotation + */(RotationAnimator?.Invoke(t) ?? 0)) * RayMath.RAD2DEG;
        Color color = ColorAnimator?.Invoke(t) ?? Color;

        Texture.Draw(pos, Pivot, scale, rot, color);
    }
}
