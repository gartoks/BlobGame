
using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Drawing;
/// <summary>
/// Class to occasionally scroll fun messages across the screen.
/// </summary>
internal sealed class TextScroller {
    private static readonly Vector2 START_POS = new Vector2(0, Application.BASE_HEIGHT * 0.85f);
    private static readonly float BASE_DISTANCE = Application.BASE_WIDTH / MathF.Cos(12.5f * RayMath.DEG2RAD);
    private static readonly Vector2 DIRECTION = new Vector2(MathF.Cos(-12.5f * RayMath.DEG2RAD), MathF.Sin(-12.5f * RayMath.DEG2RAD));

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

    private TextResource ScrollerTexts { get; set; }

    /// <summary>
    /// The next time a scroller appears.
    /// </summary>
    private float NextScrollTime { get; set; }
    private (string text, Vector2 size)? Scroller { get; set; }

    public TextScroller(float minScrollInterval, float maxScrollInterval, float scrollTime) {
        MinScrollInterval = minScrollInterval;
        MaxScrollInterval = maxScrollInterval;
        ScrollTime = scrollTime;

        Random = new Random();
    }

    internal void Load() {
        ScrollerTexts = ResourceManager.GetText("scrollers");

        NextScrollTime = Renderer.Time + MinScrollInterval;// + Random.NextSingle() * (MaxScrollInterval - MinScrollInterval);
    }

    internal void Draw() {
        if (Scroller == null) {
            if (Renderer.Time >= NextScrollTime)
                CreateScroller();

            return;
        }

        float distance = BASE_DISTANCE + Scroller.Value.size.X;
        float speed = BASE_DISTANCE / ScrollTime;

        float traversedDistance = (Renderer.Time - NextScrollTime) * speed;
        Vector2 pos = START_POS + (-Scroller.Value.size.X + (distance - traversedDistance)) * DIRECTION;

        Renderer.Font.Draw(
            Scroller.Value.text, 200,
            ResourceManager.GetColor("background"),
            pos, -12.5f);

        if (traversedDistance > distance) {
            NextScrollTime = Renderer.Time + MinScrollInterval + Random.NextSingle() * (MaxScrollInterval - MinScrollInterval);
            Scroller = null;
        }
    }

    internal void Unload() {

    }

    private void CreateScroller() {
        int idx = Random.Next(ScrollerTexts.Resource.Count);
        string scrollerText = ScrollerTexts.Resource.Values.Skip(idx).First().Trim();

        Vector2 size = Raylib.MeasureTextEx(Renderer.Font.Resource, scrollerText, 200, 200 / 16f);

        Scroller = (scrollerText, size);
    }
}
