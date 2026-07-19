fn configure_render(left_card, right_card, left_print, right_print) {
    let config = #{
        text_size: 60.0,
        text_padding_from_edge: 190,
        padding_between_cards: 20,
        text_padding_from_bottom: 80,
        text_color_r: 255,
        text_color_g: 255,
        text_color_b: 255,
        text_color_a: 255,
    };

    if left_print <= 10 && right_print <= 10 {
        config.text_color_r = 255;
        config.text_color_g = 215;
        config.text_color_b = 0;
    }

    config
}
