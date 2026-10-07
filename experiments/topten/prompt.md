Generate a static website (in a single `index.html` file) that presents a question.
Questions are sampled randomly from a list encoded in JavaScript inside the file.

Each 'question' option consists of three things:
1. the prompt, e.g. "You are a superhero with a special ability. What is your superpower?"
2. description of low score, e.g. "completely useless"
3. description of high score, e.g. "limitless power"

The website should be optimized for a mobile phone screen.
Render the prompt at the top in a larger font, and the two descriptions below it as "From <low score desc> (1) to <high score desc> (2)".
At the bottom of the screen, place a button that samples a new question from the list.

Only add the superhero example question to the list for now, I will expand it with more questions later.