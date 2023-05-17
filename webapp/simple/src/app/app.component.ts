import { Component } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { AppService } from './app.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'simple';
  loading = false;
  searchvalue = '';
  res = {
    pos: null,
    prob: null,
    pos_id: null,
    text: null,
  }

  constructor(
    private appService: AppService,
  ) {
    (window as any)['app'] = this
   }

  async search() {
    this.loading = true;
    this.res = {
      pos: null,
      prob: null,
      pos_id: null,
      text: null,
    }
    let res: any = await firstValueFrom(this.appService.getpos(this.searchvalue))
    setTimeout(() => {
      this.loading = false;
      this.res = {
        pos: res.result_pos,
        prob: res.result_prob.map((x: any) => (100*x).toFixed(1)+'%'),
        pos_id: res.result_pos_id,
        text: res.result_sent,
      }
    }, 750);
  }
}
