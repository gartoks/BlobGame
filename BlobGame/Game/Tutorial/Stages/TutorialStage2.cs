using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using System.Numerics;

namespace BlobGame.Game.Tutorial.Stages;
internal class TutorialStage2 : TutorialStage {
    private TextureResource SpeechbubbleTexture { get; set; }

    private AnimatedTexture AnimatedSpeechbubble { get; set; }

    internal override bool IsFadeInFinished => true;
    internal override bool IsFadeOutFinished => true;

    public TutorialStage2() {

    }

    internal override void Load() {
        base.Load();
        SpeechbubbleTexture = ResourceManager.GetTexture("speechbubble");

        AnimatedSpeechbubble = new AnimatedTexture(
            SpeechbubbleTexture,
            2,
            new Vector2(Application.BASE_WIDTH / 2 + 20, Application.BASE_HEIGHT / 2 + 70),
            new Vector2(0.5f, 0.5f)) {
            ScaleAnimator = t => Vector2.One + new Vector2(0.05f, 0.05f) * GetSpeechbubbleScaleT(t),
            RotationAnimator = t => MathF.PI / 128 * GetSpeechbubbleRotationT(t)
        };
    }

    internal override void Unload() {
        base.Unload();
    }

    internal override void DrawFadeIn() {
    }

    internal override void Draw() {
        AvatarTexture.Draw(new Vector2(-100, Application.BASE_HEIGHT - AvatarTexture.Resource.height / 2));

        DrawSpeechBubble();

        Renderer.Font.Draw(
            "Click to drop!",
            50,
            ResourceManager.GetColor("dark_accent"),
            new Vector2(600, Application.BASE_HEIGHT / 2 - 100));

        DrawLMBHint(750);
    }

    internal override void DrawFadeOut() {
    }

    private void DrawSpeechBubble() {
        if (AnimatedSpeechbubble.IsReady)
            AnimatedSpeechbubble.Start();

        AnimatedSpeechbubble.Draw();

        if (AnimatedSpeechbubble.IsFinished)
            AnimatedSpeechbubble.Start();
    }

    private float GetSpeechbubbleScaleT(float t) {
        return 0.5f * (MathF.Sin(MathF.Tau * t - MathF.PI / 2) + 1);
    }

    private float GetSpeechbubbleRotationT(float t) {
        return -MathF.Sin(MathF.Tau * t);
    }

}
