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

    private Func<float, Vector2>? PositionAnimator { get; init; }
    private Func<float, Vector2>? ScaleAnimator { get; init; }
    private Func<float, float>? RotationAnimator { get; init; }
    private Func<float, Color>? ColorAnimator { get; init; }

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

    public AnimatedTexture(TextureResource texture, float duration, Vector2 position, Vector2 pivot, Vector2 scale, float rotation, Color color) {
        Texture = texture;
        Position = position;
        Pivot = pivot;
        Scale = scale;
        Rotation = rotation;
        Duration = duration;
        Color = color;

        StartTime = float.MinValue;
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
        StartTime = float.MinValue;
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
        Vector2 scale = Scale * (ScaleAnimator?.Invoke(t) ?? Vector2.Zero);
        float rot = Rotation + (RotationAnimator?.Invoke(t) ?? 0);
        Color color = ColorAnimator?.Invoke(t) ?? Color;

        Texture.Draw(pos, Pivot, scale, rot, color);
    }
}
