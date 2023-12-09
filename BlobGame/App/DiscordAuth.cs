using System.Diagnostics;
using System.Dynamic;
using System.Net.Http.Json;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Web;
using Microsoft.CSharp.RuntimeBinder;

namespace BlobGame.App;
/// <summary>
/// Class for managing discord authentication
/// </summary>
static internal class DiscordAuth{

    private const string DISCORD_API_ENDPOINT = "https://discord.com/api/v10";
    private const string SCOREBOARD_API_ENDPOINT = "http://localhost:5173/api";
    private const string USER_AGENT = "Toasted! (https://github.com/gartoks/BlobGame)";

    /// <summary>
    /// The current users access and refresh tokens
    /// </summary>
    private static Tokens? _Tokens;
    /// <summary>
    /// The current users username
    /// </summary>
    public static string Username { get; private set; } = "No user";
    /// <summary>
    /// The current users user id
    /// </summary>
    public static string UserID { get; private set; } = "No user";

    /// <summary>
    /// Retrieves the username and id for the current tokens.
    /// </summary>
    internal static void UpdateUserInfo(){
        if (!IsSignedIn){
            Username = "No user";
            UserID = "No user";
            return;
        }

        var client = new HttpClient();
        var request = new HttpRequestMessage();
        request.RequestUri = new Uri(DISCORD_API_ENDPOINT + "/users/@me");
        request.Method = HttpMethod.Get;
        request.Headers.Add("Accept", "*/*");
        request.Headers.Add("User-Agent", USER_AGENT);
        request.Headers.Add("Authorization", $"Bearer {_Tokens.AccessToken}");

        var response = client.Send(request);
        var result = response.Content.ReadFromJsonAsync<JsonNode>();
        result.Wait();
        Username = result.Result["global_name"].GetValue<string>();
        UserID = result.Result["id"].GetValue<string>();
    }

    /// <summary>
    /// Indicates if a used is signed in
    /// </summary>
    internal static bool IsSignedIn { get => _Tokens != null && _Tokens.AccessToken != null && _Tokens.RefreshToken != null; }

    /// <summary>
    /// Sets the new tokens after testing their validity
    /// </summary>
    /// <param name="newTokens"></param>
    internal static void SetTokens(Tokens newTokens){
        if (IsValidToken(newTokens)){
            _Tokens = newTokens;
        }
    }
    /// <summary>
    /// Returns the current tokens and refreshes them if needed.
    /// </summary>
    /// <returns></returns>
    internal static Tokens? GetTokens(){
        if (_Tokens == null){
            return null;
        }
        if (IsValidToken(_Tokens)){
            return _Tokens;
        }

        RefreshTokens();
        return _Tokens;
    }

    /// <summary>
    /// Refreshes the access token.
    /// </summary>
    private static void RefreshTokens(){
        if (_Tokens == null){
            return;
        }

        var client = new HttpClient();
        var request = new HttpRequestMessage();
        request.RequestUri = new Uri(SCOREBOARD_API_ENDPOINT + "/game/refresh");
        request.Method = HttpMethod.Post;
        request.Headers.Add("Accept", "*/*");
        request.Headers.Add("User-Agent", USER_AGENT);
        request.Content = new FormUrlEncodedContent(new List<KeyValuePair<string, string>>(){
            new("token", _Tokens.RefreshToken)
        });

        var response = client.Send(request);
        var new_token = response.Content.ReadFromJsonAsync<JsonNode>();
        new_token.Wait();
        _Tokens = new Tokens(new_token.Result["discord_access_token"].GetValue<string>(), _Tokens.RefreshToken);
    }

    /// <summary>
    /// Tests if the given tokens are still valid and not expired.
    /// </summary>
    /// <param name="tokens">The tokens to test</param>
    /// <returns></returns>
    private static bool IsValidToken(Tokens tokens){
        var client = new HttpClient();
        var request = new HttpRequestMessage();
        request.RequestUri = new Uri(DISCORD_API_ENDPOINT + "/users/@me");
        request.Method = HttpMethod.Get;
        request.Headers.Add("Accept", "*/*");
        request.Headers.Add("User-Agent", USER_AGENT);
        request.Headers.Add("Authorization", $"Bearer {tokens.AccessToken}");

        var response = client.Send(request);
        return response.StatusCode == System.Net.HttpStatusCode.OK;
    }

    /// <summary>
    /// Signs in a user. This will open a browser and communicate with the scoreboard server.
    /// </summary>
    internal static void SignIn(){
        var client = new HttpClient();
        var request = new HttpRequestMessage();
        request.RequestUri = new Uri(SCOREBOARD_API_ENDPOINT + "/game/request_token");
        request.Method = HttpMethod.Get;

        request.Headers.Add("Accept", "*/*");
        request.Headers.Add("User-Agent", USER_AGENT);

        var response = client.Send(request);
        var result = response.Content.ReadFromJsonAsync<JsonNode>();
        result.Wait();
        var exchange_token = result.Result["exchange_token"].GetValue<string>();

        Process.Start(new ProcessStartInfo(){
            UseShellExecute = true,
            FileName = SCOREBOARD_API_ENDPOINT + "/game/auth?state=" + exchange_token,
        });

        Tokens tokens = null;

        while (tokens == null){
            Thread.Sleep(1000);

            request = new HttpRequestMessage();
            request.RequestUri = new Uri(SCOREBOARD_API_ENDPOINT + "/game/request_token?exchange_token=" + exchange_token);
            request.Method = HttpMethod.Post;

            request.Headers.Add("Accept", "*/*");
            request.Headers.Add("User-Agent", USER_AGENT);

            response = client.Send(request);
            if (response.StatusCode == System.Net.HttpStatusCode.Accepted)
                continue;

            if (response.StatusCode == System.Net.HttpStatusCode.OK){
                var result2 = response.Content.ReadFromJsonAsync<JsonNode>();
                result2.Wait();
                tokens = new Tokens(result2.Result["access_token"].GetValue<string>(), result2.Result["refresh_token"].GetValue<string>());
            }
            break;
        }

        Debug.WriteLine(tokens);
        _Tokens = tokens;
    }

    /// <summary>
    /// Signs out the user. Also revokes the old tokens.
    /// </summary>
    internal static void SignOut(){
        var client = new HttpClient();
        var request = new HttpRequestMessage();
        request.RequestUri = new Uri(SCOREBOARD_API_ENDPOINT + "/game/revoke");
        request.Method = HttpMethod.Post;
        request.Headers.Add("Accept", "*/*");
        request.Headers.Add("User-Agent", USER_AGENT);
        request.Content = new FormUrlEncodedContent(new List<KeyValuePair<string, string>>(){
            new("token", _Tokens.AccessToken)
        });

        var response = client.Send(request);
        if (response.StatusCode == System.Net.HttpStatusCode.OK){
            _Tokens = null;
        }
    }

    public record Tokens(
        string AccessToken,
        string RefreshToken
    );
}
