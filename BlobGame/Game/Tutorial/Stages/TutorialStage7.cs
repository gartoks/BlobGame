using BlobGame.App;
using BlobGame.Audio;
using BlobGame.Rendering;
using BlobGame.ResourceHandling;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Graphics.Textures;

namespace BlobGame.Game.Tutorial.Stages;
internal class TutorialStage7 : TutorialStage {
    private const float AVATAR_X = 1000;

    private Texture SpeechbubbleTexture { get; set; }

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
        SpeechbubbleTexture = ResourceManager.TextureLoader.GetResource("speechbubble");

        AnimatedSpeechbubble = new AnimatedTexture(
            SpeechbubbleTexture,
            2,
            new Vector2(GameApplication.PROJECTION_WIDTH / 2 + 20, GameApplication.PROJECTION_HEIGHT / 2 + 70),
            new Vector2(300, 300),  // TODO find size
            0,
            BASE_ZINDEX,
            new Vector2(0.5f, 0.5f)) {
            ScaleAnimator = t => Vector2.One + new Vector2(0.05f, 0.05f) * GetSpeechbubbleScaleT(t),
            RotationAnimator = t => MathF.PI / 128 * GetSpeechbubbleRotationT(t)
        };

        AnimatedAvatarFadeOut = new AnimatedTexture(
            AvatarTexture,
            0.65f,
            new Vector2(-100, GameApplication.PROJECTION_HEIGHT + AvatarTexture.Height / 2),
            new Vector2(944, 1432),
            0,
            BASE_ZINDEX,
            new Vector2(0, 1)) {
            PositionAnimator = t => new Vector2(0, GetAvatarPositionT(t) * AvatarTexture.Height / 2f)
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

        Primitives.DrawSprite(new Vector2(AVATAR_X, GameApplication.PROJECTION_HEIGHT - AvatarTexture.Height / 2), new Vector2(944, 1432), new Vector2(0.5f, 0.5f), 0, BASE_ZINDEX, AvatarTexture, Color4.White);
        DrawSpeechBubble();


        MeshFont font = Fonts.GetGuiFont(50);
        Primitives.DrawText(font, "Have fun!", ResourceManager.ColorLoader.GetResource("dark_accent"), new Vector2(600, GameApplication.PROJECTION_HEIGHT / 2 - 100), new Vector2(0.5f, 0.5f), 0, BASE_ZINDEX + 1);

        DrawLMBHint(50);
    }

    internal override void DrawFadeOut() {
        if (AnimatedAvatarFadeOut.IsReady)
            AnimatedAvatarFadeOut.Start();
        AnimatedAvatarFadeOut.Render();
    }

    private void DrawSpeechBubble() {
        if (AnimatedSpeechbubble.IsReady)
            AnimatedSpeechbubble.Start();

        AnimatedSpeechbubble.Render();

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
