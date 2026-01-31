---
layout: default
title: Posts
permalink: /posts/
---

<h1 style="color: var(--cream-beige); text-align: center; margin-bottom: 2rem;">✦ All Posts</h1>

{% for post in site.posts %}
<div class="post">
  <h2 class="post-title">
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </h2>
  <span class="post-date">{{ post.date | date_to_string }}</span>
  {{ post.excerpt }}
  {% unless post.hide_read_more %}
    <a href="{{ post.url | relative_url }}" style="font-weight: 600; color: var(--retro-pink);">Read more →</a>
  {% endunless %}</div>
{% endfor %}