using BlobGame.App;
using BlobGame.ResourceHandling;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Graphics.Textures;
using static SimpleGL.Util.Math.MathUtils;

namespace BlobGame.Rendering;
/// <summary>
/// Class to handle animated textures.
/// </summary>
internal sealed class AnimatedTexture {
    private Texture Texture { get; }

    private int ZIndex { get; }
    private Vector2 Position { get; }
    private Vector2 Scale { get; }
    private float Rotation { get; }
    private Vector2 Pivot { get; }
    private Color4 Color { get; }

    private float Duration { get; }

    internal Func<float, Vector2>? PositionAnimator { get; init; }
    internal Func<float, Vector2>? ScaleAnimator { get; init; }
    internal Func<float, float>? RotationAnimator { get; init; }
    internal Func<float, Color4>? ColorAnimator { get; init; }

    /// <summary>
    /// Time when the animation started.
    /// </summary>
    private float StartTime { get; set; }

    /// <summary>
    /// Indicator if the animation has started and finished.
    /// </summary>
    public bool IsFinished => GameApplication.RenderGameTime - StartTime >= Duration;

    /// <summary>
    /// Indicator if the animation has not yet started.
    /// </summary>
    public bool IsReady => GameApplication.RenderGameTime - StartTime < 0;

    public AnimatedTexture(string textureKey, float duration, Vector2 position, Vector2? scale, float rotation, int zIndex, Vector2 pivot, Color4? color = null)
        : this(ResourceManager.TextureLoader.GetResource(textureKey), duration, position, scale, rotation, zIndex, pivot, color) {
    }

    public AnimatedTexture(Texture texture, float duration, Vector2 position, Vector2? scale, float rotation, int zIndex, Vector2? pivot = null, Color4? color = null) {
        if (pivot == null)
            pivot = Vector2.Zero;

        if (scale == null)
            scale = Vector2.One;

        if (color == null)
            color = Color4.White;

        Texture = texture;
        Position = position;
        Pivot = pivot.Value;
        Scale = scale.Value;
        Rotation = rotation;
        Duration = duration;
        Color = color.Value;
        ZIndex = zIndex;

        StartTime = -float.MinValue;
    }

    /// <summary>
    /// Starts the animation.
    /// </summary>
    public void Start() {
        StartTime = GameApplication.RenderGameTime;
    }

    /// <summary>
    /// Resets the animation to the "not yet started" state.
    /// </summary>
    public void Reset() {
        StartTime = -float.MinValue;
    }

    public void Render() {
        float t;
        if (IsReady)
            t = 0;
        else if (IsFinished)
            t = 1;
        else
            t = (GameApplication.RenderGameTime - StartTime) / Duration;

        Vector2 pos = Position + (PositionAnimator?.Invoke(t) ?? Vector2.Zero);
        Vector2 scale = Scale * (ScaleAnimator?.Invoke(t) ?? Vector2.One);
        float rot = (/*Rotation + */(RotationAnimator?.Invoke(t) ?? Rotation)).ToDeg();
        Color4 color = ColorAnimator?.Invoke(t) ?? Color;

        Primitives.DrawSprite(pos, scale, Pivot, rot, ZIndex, Texture, color);
    }
}
