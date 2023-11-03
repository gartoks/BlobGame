using BlobGame.Audio;
using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using System.Numerics;

namespace BlobGame.Game.Tutorial.Stages;
internal class TutorialStage7 : TutorialStage {
    private const float AVATAR_X = 1000;

    private TextureResource SpeechbubbleTexture { get; set; }

    private AnimatedTexture AnimatedSpeechbubble { get; set; }
    private AnimatedTexture AnimatedAvatarFadeOut { get; set; }

    internal override bool IsFadeInFinished => true;
    internal override bool IsFadeOutFinished => AnimatedAvatarFadeOut.IsFinished;

    private bool PlayedSound { get; set; }

    public TutorialStage7() {
        PlayedSound = false;
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

        AnimatedAvatarFadeOut = new AnimatedTexture(
            AvatarTexture,
            0.65f,
            new Vector2(AVATAR_X, Application.BASE_HEIGHT + AvatarTexture.Resource.height / 2),
            new Vector2(0, 1)) {
            PositionAnimator = t => new Vector2(0, GetAvatarPositionT(t) * AvatarTexture.Resource.height / 2f)
        };
    }

    internal override void Unload() {
        base.Unload();
    }

    internal override void DrawFadeIn() {
    }

    internal override void Draw() {
        if (!PlayedSound) {
            AudioManager.PlaySound("tutorial_7");
            PlayedSound = true;
        }

        AvatarTexture.Draw(new Vector2(AVATAR_X, Application.BASE_HEIGHT - AvatarTexture.Resource.height / 2));
        DrawSpeechBubble();

        Renderer.GuiFont.Draw(
            "Have fun!",
            50,
            ResourceManager.GetColor("dark_accent"),
            new Vector2(600, Application.BASE_HEIGHT / 2 - 100));

        DrawLMBHint(50);
    }

    internal override void DrawFadeOut() {
        if (AnimatedAvatarFadeOut.IsReady)
            AnimatedAvatarFadeOut.Start();
        AnimatedAvatarFadeOut.Draw();
    }

    private void DrawSpeechBubble() {
        if (AnimatedSpeechbubble.IsReady)
            AnimatedSpeechbubble.Start();

        AnimatedSpeechbubble.Draw();

        if (AnimatedSpeechbubble.IsFinished)
            AnimatedSpeechbubble.Start();
    }

    private float GetAvatarPositionT(float t) {
        float tmp = 1.3f * t - 1;
        return 1.1f * (-(tmp * tmp) + 1);
    }

    private float GetSpeechbubbleScaleT(float t) {
        return 0.5f * (MathF.Sin(MathF.Tau * t - MathF.PI / 2) + 1);
    }

    private float GetSpeechbubbleRotationT(float t) {
        return -MathF.Sin(MathF.Tau * t);
    }

}
