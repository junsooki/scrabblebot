#include <node_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr std::array<int, 26> kLetterValues = {
    1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3,
    1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10};

struct Word {
  std::string text;
  std::array<uint8_t, 26> counts{};
  int base_score = 0;
};

std::vector<Word> g_dictionary;

int letter_index(char letter) {
  if (letter < 'A' || letter > 'Z') return -1;
  return letter - 'A';
}

int word_reward(size_t length, int base_score) {
  if (length <= 3) return base_score;
  if (length == 4) return base_score * 3 / 2;
  if (length == 5) return base_score * 2;
  if (length == 6) return base_score * 5 / 2;
  return base_score * 3;
}

double rack_leave_value(const std::array<int, 26>& rack) {
  int tile_count = 0;
  int vowels = 0;
  double value = 0;
  bool has_u = false;
  bool has_i = false;
  bool has_q = false;

  for (int i = 0; i < 26; ++i) {
    const int count = rack[i];
    if (count == 0) continue;
    const char letter = static_cast<char>('A' + i);
    tile_count += count;
    value += count * kLetterValues[i] * 1.2;
    if (letter == 'A' || letter == 'E' || letter == 'I' || letter == 'O' ||
        letter == 'U') {
      vowels += count;
    }
    if (letter == 'U') has_u = true;
    if (letter == 'I') has_i = true;
    if (letter == 'Q') has_q = true;
    value -= std::max(0, count - 1) * 0.8;
  }

  value -= std::abs(vowels - tile_count * 0.4) * 1.5;
  if (has_q && !has_u && !has_i) value -= 4;
  return value;
}

napi_value initialize_dictionary(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  if (napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr) != napi_ok ||
      argc != 1) {
    napi_throw_type_error(env, nullptr, "Expected an array of dictionary words");
    return nullptr;
  }

  bool is_array = false;
  napi_is_array(env, argv[0], &is_array);
  if (!is_array) {
    napi_throw_type_error(env, nullptr, "Dictionary must be an array");
    return nullptr;
  }

  uint32_t length = 0;
  napi_get_array_length(env, argv[0], &length);
  std::vector<Word> parsed;
  parsed.reserve(length);
  for (uint32_t i = 0; i < length; ++i) {
    napi_value element;
    napi_get_element(env, argv[0], i, &element);
    size_t utf8_length = 0;
    napi_get_value_string_utf8(env, element, nullptr, 0, &utf8_length);
    std::string text(utf8_length + 1, '\0');
    napi_get_value_string_utf8(env, element, text.data(), utf8_length + 1,
                               &utf8_length);
    text.resize(utf8_length);
    if (text.size() < 2) continue;

    Word word;
    word.text = std::move(text);
    bool valid = true;
    for (char letter : word.text) {
      const int index = letter_index(letter);
      if (index < 0 || word.counts[index] == UINT8_MAX) {
        valid = false;
        break;
      }
      ++word.counts[index];
      word.base_score += kLetterValues[index];
    }
    if (valid) parsed.push_back(std::move(word));
  }
  g_dictionary = std::move(parsed);

  napi_value result;
  napi_create_uint32(env, static_cast<uint32_t>(g_dictionary.size()), &result);
  return result;
}

napi_value choose_word(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  if (napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr) != napi_ok ||
      argc != 1) {
    napi_throw_type_error(env, nullptr, "Expected rack letters as a string");
    return nullptr;
  }

  size_t rack_length = 0;
  napi_get_value_string_utf8(env, argv[0], nullptr, 0, &rack_length);
  std::string rack_text(rack_length + 1, '\0');
  napi_get_value_string_utf8(env, argv[0], rack_text.data(), rack_length + 1,
                             &rack_length);
  rack_text.resize(rack_length);
  std::array<int, 26> rack{};
  for (char letter : rack_text) {
    const int index = letter_index(letter);
    if (index >= 0) ++rack[index];
  }

  int tile_count = 0;
  for (int count : rack) tile_count += count;

  const Word* best = nullptr;
  double best_equity = -INFINITY;
  int best_reward = -1;
  for (const Word& word : g_dictionary) {
    if (word.text.size() > static_cast<size_t>(tile_count)) continue;
    bool playable = true;
    std::array<int, 26> leave = rack;
    for (int i = 0; i < 26; ++i) {
      if (word.counts[i] > leave[i]) {
        playable = false;
        break;
      }
      leave[i] -= word.counts[i];
    }
    if (!playable) continue;

    const int reward = word_reward(word.text.size(), word.base_score);
    const double equity = reward + rack_leave_value(leave);
    if (equity > best_equity ||
        (equity == best_equity && reward > best_reward) ||
        (equity == best_equity && reward == best_reward && best != nullptr &&
         word.text.size() < best->text.size())) {
      best = &word;
      best_equity = equity;
      best_reward = reward;
    }
  }

  if (best == nullptr) {
    napi_value result;
    napi_get_null(env, &result);
    return result;
  }
  napi_value result;
  napi_create_string_utf8(env, best->text.c_str(), best->text.size(), &result);
  return result;
}

napi_value init(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
      {"initializeDictionary", nullptr, initialize_dictionary, nullptr, nullptr,
       nullptr, napi_default, nullptr},
      {"chooseWord", nullptr, choose_word, nullptr, nullptr, nullptr,
       napi_default, nullptr},
  };
  napi_define_properties(env, exports, 2, methods);
  return exports;
}

}  // namespace

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
