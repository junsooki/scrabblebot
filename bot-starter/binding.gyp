{
  "targets": [
    {
      "target_name": "scrabble_strategy",
      "sources": [ "native/strategy.cc" ],
      "defines": [ "NAPI_VERSION=8" ],
      "conditions": [
        [ "OS=='win'", {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "AdditionalOptions": [ "/std:c++17" ]
            }
          }
        }, {
          "cflags_cc": [ "-std=c++17" ],
          "xcode_settings": {
            "CLANG_CXX_LANGUAGE_STANDARD": "c++17"
          }
        } ]
      ]
    }
  ]
}
