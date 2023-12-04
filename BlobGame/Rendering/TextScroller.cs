
using BlobGame.App;
using BlobGame.ResourceHandling;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Util.Math;

namespace BlobGame.Rendering;
/// <summary>
/// Class to occasionally scroll fun messages across the screen.
/// </summary>
internal sealed class TextScroller {
    private static readonly Vector2 START_POS = new Vector2(0, GameApplication.PROJECTION_HEIGHT * 0.85f);
    private static readonly float BASE_DISTANCE = GameApplication.PROJECTION_WIDTH / MathF.Cos(12.5f.ToRad());
    private static readonly Vector2 DIRECTION = new Vector2(MathF.Cos(-12.5f.ToRad()), MathF.Sin(-12.5f.ToRad()));

    /// <summary>
    /// The time before the first scroll happens.
    /// </summary>
    private float InitialScrollInterval { get; }
    /// <summary>
    /// The minimum time between two scrollers.
    /// </summary>
    private float MinScrollInterval { get; }
    /// <summary>
    /// The maximum time between two scrollers.
    /// </summary>
    private float MaxScrollInterval { get; }
    /// <summary>
    /// The base time it takes for a scroller to traverse the screen.
    /// </summary>
    private float ScrollTime { get; }

    private Random Random { get; }

    private IReadOnlyDictionary<string, string> ScrollerTexts { get; set; }

    /// <summary>
    /// The next time a scroller appears.
    /// </summary>
    private float NextScrollTime { get; set; }
    private (string text, Vector2 size)? Scroller { get; set; }

    public TextScroller(float initialScrollInterval, float minScrollInterval, float maxScrollInterval, float scrollTime) {
        MinScrollInterval = minScrollInterval;
        MaxScrollInterval = maxScrollInterval;
        InitialScrollInterval = initialScrollInterval;
        ScrollTime = scrollTime;

        Random = new Random();
    }

    internal void Load() {
        ScrollerTexts = ResourceManager.TextLoader.GetResource("scrollers");

        NextScrollTime = GameApplication.RenderGameTime + InitialScrollInterval;
    }

    internal void Draw() {
        if (Scroller == null) {
            if (GameApplication.RenderGameTime >= NextScrollTime)
                CreateScroller();

            return;
        }

        float distance = BASE_DISTANCE + Scroller.Value.size.X;
        float speed = BASE_DISTANCE / ScrollTime;

        float traversedDistance = (GameApplication.RenderGameTime - NextScrollTime) * speed;
        Vector2 pos = START_POS + (-Scroller.Value.size.X + (distance - traversedDistance)) * DIRECTION;

        MeshFont font = Fonts.GetMainFont(200);
        Primitives.DrawText(font, Scroller.Value.text, ResourceManager.ColorLoader.GetResource("background"), pos, new Vector2(0.5f, 0.5f), -12.5f.ToRad(), 2);

        if (traversedDistance > distance) {
            NextScrollTime = GameApplication.RenderGameTime + MinScrollInterval + Random.NextSingle() * (MaxScrollInterval - MinScrollInterval);
            Scroller = null;
        }
    }

    internal void Unload() {

    }

    private void CreateScroller() {
        int idx = Random.Next(ScrollerTexts.Count);
        string scrollerText = ScrollerTexts.Values.Skip(idx).First().Trim();

        MeshFont font = Fonts.GetMainFont(200);
        Vector2 size = font.MeasureText(scrollerText);

        Scroller = (scrollerText, size);
    }
}
