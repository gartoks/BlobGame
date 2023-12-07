using BlobGame.ResourceHandling.Resources;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Drawing;
/// <summary>
/// Class to handle animated textures.
/// </summary>
internal sealed class AnimatedTexture {
    private IDrawableResource Texture { get; }

    private Vector2 Position { get; }
    private Vector2 Size { get; }
    private float Rotation { get; }
    private Vector2 Pivot { get; }
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

    public AnimatedTexture(IDrawableResource texture, float duration, Vector2 position, Vector2 size, Vector2? pivot = null, float rotation = 0, Color? color = null) {
        if (pivot == null)
            pivot = Vector2.Zero;

        if (color == null)
            color = Raylib.WHITE;

        Texture = texture;
        Position = position;
        Size = size;
        Pivot = pivot.Value;
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
        Vector2 size = Size * (ScaleAnimator?.Invoke(t) ?? Vector2.One);
        float rot = /*Rotation + */RotationAnimator?.Invoke(t) ?? Rotation;
        Color color = ColorAnimator?.Invoke(t) ?? Color;

        Texture.Draw(new Rectangle(pos.X, pos.Y, size.X, size.Y), Pivot, rot, color);
    }
}
